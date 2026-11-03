"""The processor preflight (te train --preflight): render a few bundle examples with the
pinned Muse Glimmer processor, as training will see them, before the paid smoke.

HANDOVER (M2): run it on an example and, if the system turn renders badly, write the
dataset with system_in_user=True; check first how a bare yes/no target renders. The
recipe's README: "The processor preflight and two-step training smoke cover different
failure classes; run both."

Two halves. On Modal (modal_train.preflight, CPU, no weights) render_all() runs the
processor on each example in both layouts: its token count as the cookbook counts it
(apply_chat_template with add_generation_prompt=False, tokenize=True, return_dict=True;
build_processor_length_counter in meta-oss-cookbook b30ca2c, muse-glimmer/post-training/
sft/axolotl/scripts/prepare_dataset.py), its text, and the assistant target: what the full
conversation adds to the prompt with its generation prompt. On this Mac summarize()
judges it: every example within the recipe's sequence_len, the system prompt rendered
once, the yes and no token IDs, the stop tokens (200001 and 200008, never 200007; vllm.md)
and which TE_SYSTEM_IN_USER to use.

modal_train.py copies this file into the container as te_preflight, so it imports nothing
from backend.
"""

from __future__ import annotations

import copy

STOP_IDS = (200001, 200008)  # <|end_of_text|>, <|eot|>: vllm.md "Stop tokens"
NEVER_STOP = 200007  # <|eom|> ends a message, not the turn
SPECIAL = {"<|end_of_text|>": 200001, "<|eot|>": 200008, "<|eom|>": 200007}
LAYOUTS = ("system_turn", "system_in_user")
TEXT_CHARS = 3000  # rendered text kept per example


def _text_part(messages: list[dict]) -> dict:
    user = next(m for m in messages if m["role"] == "user")
    return next(p for p in user["content"] if p.get("type") == "text")


def both_layouts(record: dict, system_prompt: str) -> dict[str, list[dict]]:
    """The record's messages with the system prompt in its own turn and folded into the
    user text (dataset.build_record writes either), whichever way the bundle has it."""
    messages = copy.deepcopy(record["messages"])
    system = system_prompt
    if messages[0]["role"] == "system":
        system = messages.pop(0)["content"]
    else:
        part = _text_part(messages)
        part["text"] = part["text"].removeprefix(f"{system_prompt}\n\n")
    folded = copy.deepcopy(messages)
    part = _text_part(folded)
    part["text"] = f"{system}\n\n{part['text']}"
    return {"system_turn": [{"role": "system", "content": system}, *messages],
            "system_in_user": folded}


def pick(records: list[dict], n: int) -> list[dict]:
    """Up to n records by id, yes and no answers taken in turn."""
    by_answer: dict[str, list[dict]] = {}
    for record in sorted(records, key=lambda r: r["id"]):
        by_answer.setdefault(record["messages"][-1]["content"], []).append(record)
    queues, out = list(by_answer.values()), []
    while len(out) < n and any(queues):
        for queue in queues:
            if queue and len(out) < n:
                out.append(queue.pop(0))
    return out


def examples(records: list[dict], n: int, system_prompt: str) -> list[dict]:
    """What modal_train.preflight renders: n records, each in both layouts."""
    return [{"id": r["id"], "answer": r["messages"][-1]["content"], "layout": layout,
             "messages": messages}
            for r in pick(records, n)
            for layout, messages in both_layouts(r, system_prompt).items()]


# -- on Modal: the processor ----------------------------------------------------------

def _ids(encoded) -> list[int]:
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def _shared(a, b) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def render(processor, messages: list[dict]) -> dict:
    """One conversation as training renders it, and its assistant target."""
    def template(msgs, prompt: bool, tokenize: bool):
        kwargs = {"tokenize": True, "return_dict": True, "return_tensors": None} if tokenize \
            else {"tokenize": False}
        return processor.apply_chat_template(msgs, add_generation_prompt=prompt, **kwargs)

    full, prompt = _ids(template(messages, False, True)), _ids(template(messages[:-1], True, True))
    text, prompt_text = template(messages, False, False), template(messages[:-1], True, False)
    tok = getattr(processor, "tokenizer", processor)
    target = full[_shared(full, prompt):]
    return {"tokens": len(full), "text": text[:TEXT_CHARS],
            "target_text": text[_shared(text, prompt_text):], "target_ids": target,
            "target_tokens": [str(t) for t in tok.convert_ids_to_tokens(target)]}


def render_all(processor, items: list[dict], generation_eos=None) -> dict:
    """render() for each item (an error is kept, not raised), and the token IDs."""
    tok = getattr(processor, "tokenizer", processor)
    renders = []
    for item in items:
        entry = {k: item[k] for k in ("id", "answer", "layout")}
        try:
            entry.update(render(processor, item["messages"]))
        except Exception as err:  # noqa: BLE001 - a template error is a finding, not a crash
            entry["error"] = f"{type(err).__name__}: {err}"[:500]
        renders.append(entry)
    vocab = tok.get_vocab()
    ids = {"yes": list(tok.encode("yes", add_special_tokens=False)),
           "no": list(tok.encode("no", add_special_tokens=False)),
           "special": {name: vocab.get(name) for name in SPECIAL},
           "eos_token_id": tok.eos_token_id}
    return {"processor": type(processor).__name__, "renders": renders, "ids": ids,
            "generation_eos": generation_eos}


# -- on this Mac: the verdict ---------------------------------------------------------

def _contains(seq: list[int], part: list[int]) -> bool:
    return bool(part) and any(seq[i:i + len(part)] == part for i in range(len(seq)))


def _system_state(renders: list[dict], system_prompt: str) -> str:
    errors = [r["error"] for r in renders if "error" in r]
    if errors:
        return f"fails: {errors[0]}"
    counts = {r["text"].count(system_prompt) for r in renders}
    if counts == {1}:
        return "renders once"
    if counts == {0}:
        return "missing from the rendered text"
    return f"renders {', '.join(map(str, sorted(counts)))} times"


def summarize(rendered: dict, *, sequence_len: int, system_prompt: str,
              bundle_system_in_user: bool) -> dict:
    """The preflight's findings and verdict from render_all's output."""
    renders = rendered["renders"]
    state = {layout: _system_state([r for r in renders if r["layout"] == layout],
                                   system_prompt) for layout in LAYOUTS}
    if state["system_turn"] == "renders once":
        folded = False
        advice = "TE_SYSTEM_IN_USER=0 (the default) works: the system turn renders once"
    elif state["system_in_user"] == "renders once":
        folded = True
        advice = (f"set TE_SYSTEM_IN_USER=1: the system turn {state['system_turn']}, folded "
                  "into the user text it renders once")
    else:
        folded = None
        advice = "neither layout renders the system prompt once: read the texts below"
    # the layout training will use: the advised one, else the bundle's
    layout = LAYOUTS[int(bundle_system_in_user if folded is None else folded)]
    used = [r for r in renders if r["layout"] == layout and "error" not in r]
    over = [f"{r['id']} ({r['tokens']} tokens)" for r in used if r["tokens"] > sequence_len]
    longest = max(used, key=lambda r: r["tokens"], default=None)
    ids, notes = rendered["ids"], []
    for answer in ("yes", "no"):
        if len(ids[answer]) != 1:
            notes.append(f"{answer!r} is {len(ids[answer])} tokens, not one: {ids[answer]}")
    targets = [{"id": r["id"], "answer": r["answer"], "text": r["target_text"],
                "tokens": r["target_tokens"],
                "answer_in_target": _contains(r["target_ids"], ids.get(r["answer"], [])),
                "ends_with_stop": bool(r["target_ids"]) and r["target_ids"][-1] in STOP_IDS}
               for r in used]
    notes += [f"{t['id']}: the target does not hold the {t['answer']!r} token on its own"
              for t in targets if not t["answer_in_target"]]
    wrong = {k: v for k, v in ids["special"].items() if v != SPECIAL[k]}
    if wrong:
        notes.append(f"special token IDs differ from vllm.md: {wrong}")
    eos = rendered.get("generation_eos")
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if eos and (not set(STOP_IDS) <= set(eos) or NEVER_STOP in eos):
        notes.append(f"generation_config eos_token_id is {eos}: vllm.md wants 200001 and "
                     "200008, never 200007")
    if folded is not None and folded != bundle_system_in_user:
        notes.append(f"this bundle was written with TE_SYSTEM_IN_USER="
                     f"{int(bundle_system_in_user)}: set TE_SYSTEM_IN_USER={int(folded)} in "
                     ".env, run te teach again for each skill (cached calls are free), then "
                     "te train again")
    ok = not over and bool(used) and folded is not None and folded == bundle_system_in_user
    return {"ok": ok, "processor": rendered["processor"], "sequence_len": sequence_len,
            "examples": len({r["id"] for r in renders}), "layout_checked": layout,
            "longest": longest and {"id": longest["id"], "tokens": longest["tokens"]},
            "over_length": over, "system_turn": state["system_turn"],
            "system_in_user": state["system_in_user"], "use_system_in_user": folded,
            "bundle_system_in_user": bundle_system_in_user, "advice": advice,
            "rendered": {lay: next((r["text"] for r in renders if r["layout"] == lay
                                    and "error" not in r), None) for lay in LAYOUTS},
            "targets": targets, "ids": ids, "generation_eos": eos or None, "notes": notes}


def format_report(report: dict) -> str:
    """The preflight as te train --preflight prints it."""
    longest = report["longest"] or {"id": "none", "tokens": 0}
    lines = [(f"processor preflight ({report['processor']}): {report['examples']} examples; "
              f"longest {longest['tokens']} tokens ({longest['id']}), sequence_len "
              f"{report['sequence_len']}")]
    lines += [f"  over sequence_len: {item}" for item in report["over_length"]]
    lines += [f"system prompt in its own turn: {report['system_turn']}",
              f"system prompt folded into the user text: {report['system_in_user']}",
              f"verdict: {report['advice']}"]
    for layout, text in report["rendered"].items():
        if text:
            lines += [f"--- rendered ({layout}) ---", text, "---"]
    for t in report["targets"][:2]:
        lines.append(f"target for {t['answer']!r} ({t['id']}): {t['text']!r} -> {t['tokens']}")
    ids = report["ids"]
    lines.append(f"token IDs: yes {ids['yes']}, no {ids['no']}, special {ids['special']}, "
                 f"generation eos {report['generation_eos']}")
    lines += [f"note: {note}" for note in report["notes"]]
    lines.append("preflight: " + ("passed" if report["ok"] else "look at the lines above"))
    return "\n".join(lines)
