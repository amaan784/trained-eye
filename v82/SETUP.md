# Setup

Everything needed to run Trained Eye and to keep building it, on the MacBook or in a
cloud session. HANDOVER.md says what to do next; this file says how to get set up.

## 1. Accounts and keys

| What | Used for | Needed | Where it goes |
|---|---|---|---|
| Meta Model API key | every Meta call (`te models`, `te api-smoke`, `te teach`) | now | `MODEL_API_KEY` in `.env`. Also send the $150 credit claim form before Nov 4. |
| Modal account and token | the W&B smoke run now, training from M2 | now | `modal setup` (or `modal token set`) on the Mac |
| Weights & Biases API key | the smoke run, training logs | now | `WANDB_API_KEY` in `.env`, and a Modal Secret named `wandb` |
| Hugging Face write token | publishing weights in M4, only after the author approves | later | `HF_TOKEN` in `.env`, and a Modal Secret named `huggingface` |
| Apple ID in Xcode | building apps onto the iPhone | now (glasses check) | Xcode, Settings, Accounts |
| Wearables Developer Center | not needed for Developer Mode builds; maybe for camera approval or release channels | only if the glasses check asks for it | wearables.developer.meta.com |

Keys never go into git (`.env` and its copies are gitignored), are never printed by the
code, and are never pasted into a chat.

## 2. The MacBook

Requirements: macOS with Xcode 26.4 or newer, Homebrew, Python 3.11 or newer (3.12
recommended), ffmpeg, git.

    xcodebuild -version                      # 26.4 or newer
    sysctl hw.memsize                        # decides Mac vs Modal serving in M2
    brew install python@3.12 ffmpeg

    git clone https://github.com/amaan784/trained-eye ~/trained-eye
    cd ~/trained-eye
    git checkout build-stages                # the finished code is at the root of this branch
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,train]"
    cp .env.example .env                     # fill in MODEL_API_KEY and WANDB_API_KEY

Modal: run `modal setup` (it opens the browser) or `modal token set` with a token from
the Modal dashboard. Then, in the dashboard under Secrets, create a secret named `wandb`
with the key `WANDB_API_KEY`. Later, for publishing, a secret named `huggingface` with
`HF_TOKEN`.

Check that everything works. On the stage branch, run these at the repository root, not
inside a v folder:

    pytest                                   # all offline; none calls a paid API
    ruff check .
    te spend                                 # Meta API spend so far
    te models                                # free: model IDs for your key, and any missing
    te api-smoke                             # one Muse Spark call, at most about $0.0024
    modal run backend/training/wandb_smoke.py   # CPU only; prints the W&B run URL
    te serve                                 # then open http://<mac-ip>:8765/health on the phone

If macOS asks whether Python may accept incoming connections when `te serve` starts,
allow it.

## 3. The glasses and the iPhone

`docs/glasses-check.md` has every step: Developer Mode in the Meta AI app, Developer
Mode on the iPhone, building Meta's CameraAccess sample, the camera stream, the glasses
mic, the speakers, phone-to-Mac reachability, and the Mock Device Kit.

## 4. A cloud session

A cloud session can write and test code, run offline commands and push to GitHub. It
cannot run Xcode, use the glasses or see files on the Mac.

**Network.** The session that built M0 and M1 had these hosts denied: dev.meta.ai,
wearables.developer.meta.com, ml-explore.github.io and PyPI. GitHub git reads worked. To
make live calls or install packages from a cloud session, open the cloud environment's
settings (the environment menu in the session's title bar, then Edit), go to Network
access, and either choose a broader level or add allowed domains with "Allow package
managers" ticked. The hosts the project talks to include api.meta.ai, dev.meta.ai,
wearables.developer.meta.com, ml-explore.github.io, api.wandb.ai, api.modal.com and
huggingface.co.

**Keys.** In the same settings, add environment variables named `MODEL_API_KEY`,
`WANDB_API_KEY`, `HF_TOKEN`, and for Modal `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
`backend/config.py` reads the environment before `.env`. A new session picks them up.

**If PyPI is still blocked.** The previous session's image had pydantic, starlette,
uvicorn, numpy and pillow installed system-wide, and pytest, httpx and ruff unpacked in
the uv cache. This made a working environment without downloading anything:

    uv venv --system-site-packages ~/venv
    SITE=$(~/venv/bin/python -c "import site; print(site.getsitepackages()[0])")
    for p in pytest pluggy iniconfig httpx httpcore pygments; do
      dirname "$(find ~/.cache/uv/archive-v0 -maxdepth 2 -iname "${p}-*.dist-info" | head -1)"
    done > "$SITE/cached_wheels.pth"
    ln -sf "$(find ~/.cache/uv/archive-v0 -path '*ruff*/scripts/ruff' | head -1)" ~/venv/bin/ruff

Other quirks seen there:

- `pip install -e .` failed for every package with `AttributeError: install_layout` (the
  image's Debian setuptools). Use `python -m backend.cli <command>` instead of `te`.
- `pyproject.toml` already passes `-q` to pytest. Adding another `-q` hides the summary
  line with the test count.
- ffmpeg was installed, so video tests ran.

## 5. Git and the stage branch

- Work happens on the branch `build-stages`. Its root is the finished project; the folders
  v1 to vN are the same work in cumulative stages, and `VERSIONS.md` explains them.
  `stagetools/` holds the scripts that build and check the stages. VERSIONS.md, the v
  folders and stagetools/ exist only on this branch.
- The author copies each stage folder onto `main` and commits it there:
  `rsync -a ~/te-stages/vN/ ./`, then `git add -A` and `git commit` with the summary and
  description from VERSIONS.md. Sessions never push to `main`.
- Commits a session makes on `build-stages` use the author's identity for both author and
  committer, no signature, and no tool or AI attribution of any kind:

      GIT_AUTHOR_NAME="Amaan Sheikh" \
      GIT_AUTHOR_EMAIL="100550299+amaan784@users.noreply.github.com" \
      GIT_COMMITTER_NAME="Amaan Sheikh" \
      GIT_COMMITTER_EMAIL="100550299+amaan784@users.noreply.github.com" \
      git -c commit.gpgsign=false commit -F message.txt
      git push -u origin build-stages

  A cloud session's own git config may set another name and a signing key; the variables
  and the `-c` flag above override both.
