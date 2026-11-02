// The playground page: recorded clips, how each skill was built, teach it yourself and
// the results. Every URL is relative, so the site also works under a path prefix.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const money = (v) => "$" + Number(v || 0).toFixed(2);
  const NS = "http://www.w3.org/2000/svg";

  async function getJSON(url) {
    const r = await fetch(url, { credentials: "same-origin" });
    if (!r.ok) throw new Error(url + " answered " + r.status);
    return r.json();
  }
  function node(tag, cls, words) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (words != null) n.textContent = words;
    return n;
  }
  function chips(box, names, open) { // one button per name; the first one opens
    box.textContent = "";
    names.forEach((name, i) => {
      const b = node("button", "", name);
      b.type = "button";
      b.addEventListener("click", () => {
        for (const other of box.children) other.classList.toggle("on", other === b);
        Promise.resolve(open(name)).catch((err) => console.error(err));
      });
      box.appendChild(b);
      if (i === 0) b.click();
    });
  }
  function failed(box, err) {
    box.hidden = false;
    box.textContent = "Could not load this part of the page (" + err.message + ").";
  }

  // -- recorded clips and the live timeline ------------------------------------------
  const video = $("clip-video");
  let timeline = null;
  for (const ev of ["timeupdate", "seeked", "play"]) {
    video.addEventListener(ev, () => timeline && timeline.update());
  }
  video.addEventListener("loadedmetadata", () => timeline && timeline.redraw());
  let resizing = 0;
  window.addEventListener("resize", () => {
    cancelAnimationFrame(resizing);
    resizing = requestAnimationFrame(() => timeline && timeline.redraw());
  });

  async function loadClips() {
    const { clips } = await getJSON("api/clips");
    $("clips-empty").hidden = clips.length > 0;
    chips($("clip-list"), clips.map((c) => c.name), openClip);
  }
  async function openClip(name) {
    const clip = await getJSON("api/clips/" + encodeURIComponent(name));
    $("clip-view").hidden = false;
    video.hidden = !clip.video;
    $("clip-novideo").hidden = !!clip.video;
    if (clip.video) video.src = clip.video; else video.removeAttribute("src");
    $("clip-alert").hidden = true;
    const names = Object.keys(clip.skills);
    timeline = null;
    $("clip-skills").hidden = names.length < 2;
    if (!names.length) {
      $("timeline").textContent = "No scored frames in this clip's log.";
      $("clip-readout").textContent = "";
      return;
    }
    chips($("clip-skills"), names, (skill) => {
      timeline = Timeline.mount($("timeline"), clip, skill, {
        video: clip.video ? video : null, banner: $("clip-alert"), readout: $("clip-readout"),
      });
    });
  }

  // -- how each skill was built ------------------------------------------------------
  let wallTiles = [];
  $("wall-replay").addEventListener("click", () => Wall.render($("wall"), wallTiles));

  async function loadSkills() {
    const { skills } = await getJSON("api/skills");
    $("skills-empty").hidden = skills.length > 0;
    chips($("skill-list"), skills.map((s) => s.name), openSkill);
  }
  async function openSkill(name) {
    const s = await getJSON("api/skills/" + encodeURIComponent(name));
    $("skill-view").hidden = false;
    $("skill-readback").textContent = s.read_back;
    const strip = $("keyframes");
    strip.textContent = "";
    s.keyframes.forEach((k, i) => {
      const img = node("img");
      img.src = k.url;
      img.loading = "lazy";
      img.alt = "Keyframe " + (i + 1) + (k.spark_label ? ", Spark said " + k.spark_label : "");
      img.title = img.alt;
      strip.appendChild(img);
    });
    drawOutline(s.roi);
    wallTiles = s.wall;
    Wall.render($("wall"), wallTiles);
    const st = s.label_stats;
    let words = st && st.total ? `Spark agreed with what ${st.kept} of the ${st.total} ` +
      `generated images were made to show` +
      (st.agreement != null ? ` (agreement ${st.agreement})` : "") +
      "; only those are used for training. " : "";
    words += `Showing ${s.wall.length} of ${s.wall_total}. ` + Wall.summary(s.wall);
    $("wall-summary").textContent = words;
  }
  function drawOutline(roi) {
    const box = $("outline");
    box.textContent = "";
    const f = roi.frame;
    if (!f) {
      box.appendChild(node("p", "empty", roi.note || "SAM found no box on any keyframe."));
      return;
    }
    const wrap = node("div", "outline");
    const img = node("img");
    img.src = f.url;
    img.alt = `Teaching frame with the box SAM 3.1 found for "${roi.prompt}"`;
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${f.width} ${f.height}`);
    svg.setAttribute("preserveAspectRatio", "none");
    const [x1, y1, x2, y2] = f.box;
    const rect = document.createElementNS(NS, "rect");
    const attrs = { x: x1, y: y1, width: x2 - x1, height: y2 - y1, rx: 6,
      "vector-effect": "non-scaling-stroke" };
    for (const k in attrs) rect.setAttribute(k, attrs[k]);
    svg.appendChild(rect);
    wrap.append(img, svg);
    box.append(wrap, node("p", "muted", `SAM 3.1 looked for "${roi.prompt}" and found it on ` +
      `${roi.with_box} of ${roi.frames} keyframes` +
      (f.score != null ? `; score on this one ${f.score.toFixed(2)}.` : ".")));
  }

  // -- teach it yourself -------------------------------------------------------------
  const say = (words, bad) => {
    $("teach-status").textContent = words;
    $("teach-status").classList.toggle("bad", !!bad);
  };
  const busy = (on) => { $("teach-go").disabled = on; };

  async function loadStatus() {
    const s = await getJSON("api/status");
    $("teach-cost").textContent = s.estimate_usd == null ? "" : `A run costs at most about ` +
      `${money(s.estimate_usd)} before cache hits. Each visitor may spend ` +
      `${money(s.visitor_cap_usd)} a day, and the playground ${money(s.daily_cap_usd)} ` +
      "a day in all.";
    $("teach-budget").textContent = s.live ? `You have ${money(s.visitor_left_usd)} left ` +
      `today; the playground has ${money(s.day_left_usd)}.` : "";
    return s;
  }
  async function loadKeyframes() {
    const { keyframes } = await getJSON("api/teach/keyframes");
    const box = $("teach-keyframes");
    box.textContent = "";
    if (!keyframes.length) {
      box.appendChild(node("p", "muted", "No keyframes are provided here yet."));
    }
    $("teach-go").disabled = !keyframes.length;
    keyframes.forEach((k, i) => {
      const label = node("label"), input = node("input"), img = node("img");
      input.type = "radio";
      input.name = "keyframe";
      input.value = k.id;
      input.checked = i === 0;
      img.src = k.url;
      img.alt = "Keyframe " + (i + 1);
      label.append(input, img);
      box.appendChild(label);
    });
  }
  async function showExamples() {
    const { examples } = await getJSON("api/examples");
    const box = $("examples");
    box.textContent = "";
    $("teach-examples").hidden = false;
    if (!examples.length) box.appendChild(node("p", "empty", "No precomputed examples yet."));
    for (const ex of examples) {
      const wall = node("div", "wall");
      box.append(node("h3", "", `"${ex.request}"`), node("p", "readback", ex.read_back),
        node("p", "muted", Wall.summary(ex.variations)), wall);
      Wall.render(wall, ex.variations);
    }
  }

  $("teach-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const condition = $("condition").value.trim();
    const picked = document.querySelector('input[name="keyframe"]:checked');
    if (!condition) return say("Type a condition first.", true);
    if (!picked) return say("Pick a keyframe first.", true);
    busy(true);
    say("Checking the policy, then asking Muse Wonder.");
    $("teach-result").hidden = true;
    $("teach-wall").textContent = ""; // nothing of the last run stays on screen
    $("teach-summary").textContent = "";
    let r, body;
    try {
      r = await fetch("api/teach", { method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ condition, keyframe: picked.value }) });
      body = await r.json();
    } catch (err) {
      busy(false);
      return say("The playground could not be reached.", true);
    }
    if (body.refused) {
      showRun({ read_back: body.read_back });
      say("Refused by the policy check before any call, so it cost nothing.");
    } else if (body.unavailable) {
      say(body.message + " Precomputed examples are shown below.", true);
      showExamples().catch(() => {});
    } else if (!r.ok || !body.job) {
      say(body.error || "Something went wrong.", true);
    } else {
      await follow(body.job);
    }
    busy(false);
    loadStatus().catch(() => {});
  });

  function showRun(v) {
    $("teach-result").hidden = false;
    $("teach-readback").textContent = v.read_back || "";
    const spec = $("teach-spec");
    spec.textContent = "";
    const rows = v.spec ? [["Looks at", v.spec.look_at], ["Fires when", v.spec.positive],
      ["Not when", v.spec.negatives.join("; ")], ["Says", v.spec.alert]] : [];
    for (const [k, val] of rows) spec.append(node("dt", "", k), node("dd", "", val));
  }

  async function follow(job) {
    const wall = $("teach-wall"), all = [];
    wall.textContent = "";
    $("teach-summary").textContent = "";
    showRun({});
    for (let seen = 0; ;) {
      let v;
      try { v = await getJSON(`api/teach/${encodeURIComponent(job)}?since=${seen}`); } catch (err) {
        return say("Lost track of the run (" + err.message + ").", true);
      }
      if (v.read_back && !$("teach-readback").textContent) showRun(v);
      for (const t of v.variations) { Wall.add(wall, t); all.push(t); }
      seen += v.variations.length;
      $("teach-summary").textContent = Wall.summary(all);
      if (v.done) {
        if (v.stopped) return say(v.stopped + ` Spent ${money(v.cost_usd)}.`, true);
        if (v.refused) return say(`Muse Wonder refused it. Spent ${money(v.cost_usd)}.`);
        return say(`Done: ${v.count} variations, live, for ${money(v.cost_usd)}.`);
      }
      if (v.read_back) say(`Muse Image and Muse Spark at work: ${v.count} of ${v.planned} done.`);
      await new Promise((ok) => setTimeout(ok, 1200));
    }
  }

  // -- results -----------------------------------------------------------------------
  async function loadResults() {
    const r = await getJSON("api/results");
    const box = $("charts");
    for (const c of r.charts) {
      const fig = node("figure"), img = node("img");
      img.src = c.url;
      img.alt = c.name;
      fig.append(img, node("figcaption", "", c.name));
      box.appendChild(fig);
    }
    if (r.results_md) {
      $("results-md").textContent = r.results_md;
      $("results-md").hidden = false;
    }
    $("results-empty").hidden = r.charts.length > 0 || !!r.results_md;
  }

  loadClips().catch((err) => failed($("clips-empty"), err));
  loadSkills().catch((err) => failed($("skills-empty"), err));
  loadResults().catch((err) => failed($("results-empty"), err));
  loadKeyframes().catch((err) => say("Could not load the keyframes (" + err.message + ").", true));
  loadStatus().then((s) => {
    if (s.live && s.visitor_left_usd >= s.estimate_usd && s.day_left_usd >= s.estimate_usd) return;
    say((s.message || "The live budget for today is used up.") +
      " Precomputed examples are shown below.", !s.live);
    return showExamples();
  }).catch((err) => say("Could not load the playground's status (" + err.message + ").", true));
})();
