// The playground page: watch a clip, how it learned, teach it yourself, the results.
// Every URL is relative, so the site also works under a path prefix.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const node = Film.node;
  const money = (v) => "$" + Number(v || 0).toFixed(2);
  const enc = encodeURIComponent;
  let publicCopy = false; // the public copy shows no generated images (api/status)
  const shownBelow = () => publicCopy ? "" : " Examples made earlier are shown below.";

  async function getJSON(url) {
    const r = await fetch(url, { credentials: "same-origin" });
    if (!r.ok) throw new Error(url + " answered " + r.status);
    return r.json();
  }
  // Text buttons, one per name; the first opens. Hidden when there is only one.
  function choices(box, names, label, open) {
    box.textContent = "";
    box.hidden = names.length < 2;
    names.forEach((name, i) => {
      const b = node("button", "", label(name));
      b.type = "button";
      b.addEventListener("click", () => {
        for (const other of box.children) other.setAttribute("aria-pressed", String(other === b));
        Promise.resolve(open(name)).catch((err) => console.error(err));
      });
      box.appendChild(b);
      if (i === 0) b.click();
    });
  }
  const plain = (name) => name.replace(/[_-]+/g, " ");
  function failed(box, err) {
    box.hidden = false;
    box.textContent = "This part of the page did not load (" + err.message + "). Reload to try again.";
  }
  const once = (el, ev) => new Promise((ok) => el.addEventListener(ev, ok, { once: true }));

  // -- watch a clip: the video above its film, the ring when the alert fires ---------
  const video = $("clip-video");
  let hero = null;
  for (const ev of ["timeupdate", "seeked", "play", "loadedmetadata"]) {
    video.addEventListener(ev, () => update());
  }
  window.addEventListener("resize", () => hero && place(true));

  async function loadClips() {
    const { clips } = await getJSON("api/clips");
    $("clips-empty").hidden = clips.length > 0;
    choices($("clip-list"), clips.map((c) => c.name), plain, openClip);
  }

  async function openClip(name) {
    const clip = await getJSON("api/clips/" + enc(name));
    $("clip-view").hidden = false;
    const names = Object.keys(clip.skills);
    const skill = names.find((k) => clip.skills[k].alerts.length) || names[0];
    hero = null;
    $("spoken").hidden = true;
    if (!skill) {
      $("hero-film").textContent = "";
      $("clip-readout").textContent = "This clip's log has no scored frames.";
      return;
    }
    const s = clip.skills[skill];
    video.hidden = !clip.video;
    if (clip.video) video.src = clip.video; else video.removeAttribute("src");
    const points = s.points.slice().sort((a, b) => a[0] - b[0]);
    const alert = s.alerts[0] || null;
    // about 40 frames over the clip, and always the frame where the alert fired
    const step = Math.max(1, Math.round(clip.duration / 40));
    const picked = new Map();
    for (let t = 0; t <= clip.duration + 1e-6; t += step) {
      const near = points.reduce((a, b) => Math.abs(b[0] - t) < Math.abs(a[0] - t) ? b : a, points[0]);
      picked.set(near[0], near);
    }
    if (alert) picked.set(alert.t, [alert.t, alert.p]);
    const frames = [...picked.values()].sort((a, b) => a[0] - b[0]);
    const canvases = frames.map(() => node("canvas"));
    const cells = frames.map(([t, p], i) => ({
      canvas: canvases[i], edge: Film.time(t) + " " + p.toFixed(2),
      title: `${Film.time(t)}: score ${p.toFixed(2)}`,
    }));
    const { film, track } = Film.strip(cells, { bleed: false }); // #hero-film bleeds
    $("hero-film").replaceChildren(film);
    const cellEls = [...track.querySelectorAll(".cell")];
    cellEls.forEach((el, i) => el.addEventListener("click", () => {
      if (clip.video) video.currentTime = frames[i][0];
    }));
    $("spoken-line").textContent = alert ? alert.text : "";
    $("spoken-when").textContent = alert && s.hold_s ?
      `Spoken on the glasses once this had lasted ${s.hold_s} seconds.` : "";
    hero = { frames, film, track, cellEls, alert, threshold: s.threshold, ringed: false,
             alertIndex: alert ? frames.findIndex((f) => f[0] === alert.t) : -1 };
    update();
    if (clip.video) thumbs(clip.video, frames.map((f) => f[0]), canvases).catch(() => {});
  }

  // Each frame of the film is drawn from the clip itself, by a second, hidden player.
  async function thumbs(src, times, canvases) {
    const v = node("video");
    v.muted = true;
    v.playsInline = true;
    v.preload = "auto";
    v.src = src;
    await once(v, "loadeddata");
    for (let i = 0; i < times.length; i++) {
      v.currentTime = Math.min(times[i], Math.max(0, v.duration - 0.05));
      await once(v, "seeked");
      const c = canvases[i], w = 240, h = 320;
      c.width = w;
      c.height = h;
      const s = Math.max(w / v.videoWidth, h / v.videoHeight);
      c.getContext("2d").drawImage(v, (w - v.videoWidth * s) / 2, (h - v.videoHeight * s) / 2,
        v.videoWidth * s, v.videoHeight * s);
    }
  }

  function current() {
    const t = video.currentTime || 0;
    let i = 0;
    while (i + 1 < hero.frames.length && hero.frames[i + 1][0] <= t + 0.05) i++;
    return { t, i };
  }

  // The current frame sits a third of the way across; the film moves under it.
  function place(now) {
    const { i } = current();
    const el = hero.cellEls[i];
    const room = hero.film.clientWidth;
    const shift = Math.max(0, Math.min(el.offsetLeft - room / 3,
                                       hero.track.scrollWidth - room));
    if (now) hero.track.style.transition = "none";
    hero.track.style.transform = `translateX(${-shift}px)`;
    if (now) requestAnimationFrame(() => { hero.track.style.transition = ""; });
  }

  function update() {
    if (!hero) return;
    const { t, i } = current();
    place(false);
    const p = hero.frames[i][1];
    $("clip-readout").textContent = `${Film.time(t)}: the model's score ${p.toFixed(2)}` +
      (hero.threshold != null ? `, against a threshold of ${hero.threshold.toFixed(2)}.` : ".");
    const fired = hero.alert && t + 0.05 >= hero.alert.t;
    if (fired && !hero.ringed) {
      Film.ring(hero.track, hero.cellEls[hero.alertIndex]);
      $("spoken").hidden = false;
      hero.ringed = true;
    } else if (!fired && hero.ringed) {
      Film.clearRing(hero.track);
      $("spoken").hidden = true;
      hero.ringed = false;
    }
  }

  // -- how it learned: the teaching frames with SAM's boxes, and the four steps -------
  async function loadSkills() {
    const { skills } = await getJSON("api/skills");
    $("skills-empty").hidden = skills.length > 0;
    choices($("skill-list"), skills.map((s) => s.name), plain, openSkill);
  }

  async function openSkill(name) {
    const s = await getJSON("api/skills/" + enc(name));
    $("skill-view").hidden = false;
    $("learned-title").textContent = "How it learned this " + name.split("_")[0];
    const kfs = s.keyframes;
    const known = kfs.filter((k) => k.t != null).map((k) => Math.floor(k.t));
    const tenths = new Set(known).size < known.length;
    const rows = [];
    for (let i = 0; i < kfs.length; i += 12) {
      rows.push(Film.strip(kfs.slice(i, i + 12).map((k, j) => ({
        src: k.url, box: k.box, size: k.size,
        edge: k.t != null ? Film.time(k.t, tenths) : "",
        alt: `Teaching frame ${i + j + 1}` + (k.box ? `, where SAM 3.1 found the ${s.target}` : ""),
      })), {}).film);
    }
    $("sheet").replaceChildren(...rows);
    const boxed = kfs.filter((k) => k.box).length;
    $("sheet-note").textContent = `${kfs.length} frames kept from the teaching clip` +
      (publicCopy ? ", faces blurred" : "") +
      (boxed ? `, with SAM's boxes in red on the ${boxed} where it found the ${s.target}.` : ".");
    const steps = $("steps");
    steps.textContent = "";
    steps.hidden = !s.steps;
    for (const x of s.steps || []) {
      const li = node("li");
      li.append(node("b", "", x.lead));
      if (x.quote) li.append(" ", node("q", "", x.quote));
      li.append(x.rest);
      steps.appendChild(li);
    }
  }

  // -- teach it yourself -------------------------------------------------------------
  // the time so far, ticking once a second outside the status line (screen readers hear
  // the status line when it changes, not every tick)
  let teachStart = 0, teachTimer = 0;
  const tick = () => {
    const s = Math.floor((Date.now() - teachStart) / 1000);
    $("teach-clock").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")} so far`;
  };
  const say = (words, bad) => {
    $("teach-status").textContent = words;
    $("teach-status").classList.toggle("bad", !!bad);
  };
  // the bar: a share done (0 to 1), or null for "working, no count yet"
  const progress = (share) => {
    const bar = $("teach-progress");
    bar.hidden = false;
    bar.classList.toggle("waiting", share == null);
    bar.firstElementChild.style.width = share == null ? "" : `${Math.round(share * 100)}%`;
  };
  const busy = (on) => {
    $("teach-go").disabled = on;
    $("teach-go").textContent = on ? "Teaching\u2026" : "Teach it";
    clearInterval(teachTimer);
    $("teach-clock").hidden = !on;
    $("teach-progress").hidden = !on;
    if (on) {
      teachStart = Date.now();
      tick();
      teachTimer = setInterval(tick, 1000);
      progress(null);
    }
  };

  async function loadStatus() {
    const s = await getJSON("api/status");
    publicCopy = !!s.public;
    $("teach-cost").textContent = s.estimate_usd == null ? "" : `A run costs at most about ` +
      `${money(s.estimate_usd)}. Each visitor may spend ${money(s.visitor_cap_usd)} a day, ` +
      `and the playground ${money(s.daily_cap_usd)} a day in all.`;
    $("teach-budget").textContent = s.live ? `You have ${money(s.visitor_left_usd)} left ` +
      `today; the playground has ${money(s.day_left_usd)}.` : "";
    return s;
  }

  async function loadKeyframes() {
    const { keyframes } = await getJSON("api/teach/keyframes");
    const box = $("teach-keyframes");
    box.textContent = "";
    $("teach-go").disabled = !keyframes.length;
    if (!keyframes.length) {
      box.appendChild(node("p", "empty", "No frames are offered here yet."));
      return;
    }
    const cells = keyframes.map((k, i) => {
      const input = node("input");
      input.type = "radio";
      input.name = "keyframe";
      input.value = k.id;
      input.checked = i === 0;
      input.setAttribute("aria-label", "Frame " + (i + 1));
      return { input, src: k.url, alt: "Frame " + (i + 1), edge: "" };
    });
    box.appendChild(Film.strip(cells, {}).film);
  }

  async function showExamples() {
    const { examples, note } = await getJSON("api/examples");
    const box = $("examples");
    box.textContent = "";
    $("teach-examples").hidden = false;
    if (!examples.length) box.appendChild(node("p", "empty", note || "No examples yet."));
    for (const ex of examples) {
      const sheet = node("div");
      const ask = node("p", "readback measure");
      ask.append(node("q", "", ex.request));
      box.append(ask, node("p", "measure", ex.read_back),
        node("p", "small soft measure", Wall.summary(ex.variations)), sheet);
      Wall.render(sheet, ex.variations);
    }
  }

  $("teach-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const condition = $("condition").value.trim();
    const picked = document.querySelector('input[name="keyframe"]:checked');
    if (!condition) return say("Write what it should watch for first.", true);
    if (!picked) return say("Pick a frame first.", true);
    busy(true);
    say("Checking the policy, then asking Muse Wonder.");
    $("teach-result").hidden = true;
    $("teach-sheet").textContent = ""; // nothing of the last run stays on screen
    $("teach-summary").textContent = "";
    let r, body;
    try {
      r = await fetch("api/teach", { method: "POST", credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ condition, keyframe: picked.value }) });
      body = await r.json();
    } catch (err) {
      busy(false);
      return say("The playground did not answer. Check your connection and try again.", true);
    }
    if (body.refused) {
      showRun({ read_back: body.read_back });
      say("The policy check refused it before any call, so it cost nothing.");
    } else if (body.unavailable) {
      say(body.message + shownBelow(), true);
      showExamples().catch(() => {});
    } else if (!r.ok || !body.job) {
      say(body.error || "Something went wrong. Try again in a minute.", true);
    } else {
      await follow(body.job);
    }
    busy(false);
    loadStatus().catch(() => {});
  });

  function showRun(v) {
    $("teach-result").hidden = false;
    $("teach-readback").textContent = v.read_back || "";
  }

  async function follow(job) {
    const sheet = $("teach-sheet"), all = [];
    sheet.textContent = "";
    showRun({});
    for (let seen = 0; ;) {
      let v;
      try { v = await getJSON(`api/teach/${enc(job)}?since=${seen}`); } catch (err) {
        return say("Lost track of the run (" + err.message + "). Reload to see what it made.", true);
      }
      if (v.read_back && !$("teach-readback").textContent) showRun(v);
      for (const t of v.variations) { Wall.add(sheet, t); all.push(t); }
      seen += v.variations.length;
      $("teach-summary").textContent = Wall.summary(all);
      if (v.done) {
        if (v.stopped) return say(v.stopped + ` It spent ${money(v.cost_usd)}.`, true);
        if (v.refused) return say(`Muse Wonder refused it. It spent ${money(v.cost_usd)}.`);
        return say(`Done: ${v.count} variations, made live for ${money(v.cost_usd)}.`);
      }
      if (v.read_back) {
        say(`Muse Image is making the variations and Muse Spark is checking them: ${v.count} ` +
            `of ${v.planned} done.`);
        progress(v.planned ? v.count / v.planned : null);
      }
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
      img.alt = c.caption || c.name;
      fig.append(img, node("figcaption", "", c.caption || c.name));
      box.appendChild(fig);
    }
    if (r.results_md) {
      $("results-md").textContent = r.results_md;
      $("results-all").hidden = false;
    }
    $("results-empty").hidden = r.charts.length > 0 || !!r.results_md;
  }

  loadClips().catch((err) => failed($("clips-empty"), err));
  loadResults().catch((err) => failed($("results-empty"), err));
  loadKeyframes().catch((err) => say("The frames did not load (" + err.message + "). Reload to try again.", true));
  const status = loadStatus();
  status.catch(() => {}).finally(() => loadSkills().catch((err) => failed($("skills-empty"), err)));
  status.then((s) => {
    if (s.live && s.visitor_left_usd >= s.estimate_usd && s.day_left_usd >= s.estimate_usd) return;
    say((s.message || "Today's live budget is used up.") + shownBelow(), !s.live);
    return showExamples();
  }).catch((err) => say("The playground's status did not load (" + err.message + ").", true));
})();
