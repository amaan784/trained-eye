// The live timeline (signature visual 2): the student's yes-probability for each scored
// frame as a line, the threshold dashed, the true ranges shaded, and an alert marker with
// the spoken text. A playhead follows the video; hover shows a frame's score, click seeks.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const H = 200, M = { l: 38, r: 14, t: 36, b: 26 };
  const SHOW_ALERT_S = 5;

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function text(parent, x, y, words, anchor, cls) {
    const node = el("text", { x, y, "text-anchor": anchor || "start" }, parent);
    if (cls) node.setAttribute("class", cls);
    node.textContent = words;
    return node;
  }
  function clock(s) {
    const m = Math.floor(s / 60), r = Math.floor(s % 60);
    return m + ":" + String(r).padStart(2, "0");
  }
  function tickStep(span, px) {
    const target = span / Math.max(2, Math.floor(px / 80));
    for (const s of [1, 2, 5, 10, 15, 30, 60, 120, 300, 600]) if (s >= target) return s;
    return 1200;
  }
  function latest(points, t) { // the last scored frame at or before t
    let found = null;
    for (const p of points) { if (p[0] > t + 1e-6) break; found = p; }
    return found;
  }

  function mount(box, clip, name, opts) {
    const skill = clip.skills[name];
    const video = opts.video, banner = opts.banner, readout = opts.readout;
    const th = skill.threshold;
    let svg, cursor, x, y, W = 0, hover = null;

    function span() {
      const d = video && isFinite(video.duration) ? video.duration : 0;
      return Math.max(clip.duration, d, 1);
    }
    function now() { return video ? video.currentTime || 0 : 0; }

    function draw() {
      W = Math.max(280, Math.round(box.clientWidth || 640));
      box.textContent = "";
      const T = span(), iw = W - M.l - M.r, ih = H - M.t - M.b;
      x = (t) => M.l + (t / T) * iw;
      y = (p) => M.t + (1 - p) * ih;
      svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
        "aria-label": `Yes-probability of ${name} over the clip, threshold dashed, ` +
          "alerts marked" }, box);
      clip.true_ranges.forEach((r, i) => {
        el("rect", { class: "truth", x: x(r[0]), y: M.t, height: ih,
          width: Math.max(2, x(r[1]) - x(r[0])) }, svg);
        if (i === 0) text(svg, x(r[0]) + 4, M.t + ih - 6, "condition true");
      });
      for (const v of [0, 0.5, 1]) {
        el("line", { class: "grid", x1: M.l, x2: W - M.r, y1: y(v), y2: y(v) }, svg);
        text(svg, M.l - 6, y(v) + 4, String(v), "end");
      }
      const dt = tickStep(T, iw);
      for (let t = 0; t <= T + 1e-6; t += dt) text(svg, x(t), H - 8, clock(t), "middle");
      if (th != null) {
        el("line", { class: "threshold", x1: M.l, x2: W - M.r, y1: y(th), y2: y(th) }, svg);
        text(svg, W - M.r, y(th) - 6, "threshold " + th.toFixed(2), "end");
      }
      let d = "", pen = false;
      for (const [t, p] of skill.points) {
        if (p == null) { pen = false; continue; } // the student failed on this frame: a gap
        d += (pen ? "L" : "M") + x(t).toFixed(1) + " " + y(p).toFixed(1);
        pen = true;
      }
      el("path", { class: "line", d }, svg);
      const rowEnds = [-Infinity, -Infinity]; // the right edge of each row's last label
      for (const a of skill.alerts) {
        const g = el("g", { class: "alert" }, svg), ax = x(a.t), right = ax > W * 0.6;
        el("line", { x1: ax, x2: ax, y1: M.t - 6, y2: M.t + ih }, g);
        if (a.p != null) el("circle", { cx: ax, cy: y(a.p), r: 4.5 }, g);
        const label = text(g, ax + (right ? -6 : 6), 13, `"${a.text}"`,
          right ? "end" : "start");
        const w = label.getComputedTextLength(), left = right ? ax - 6 - w : ax + 6;
        // the first row that is free there, else the row that ends first
        let row = rowEnds.findIndex((end) => end + 8 <= left);
        if (row < 0) row = rowEnds[0] <= rowEnds[1] ? 0 : 1;
        label.setAttribute("y", row ? 28 : 13);
        rowEnds[row] = left + w;
      }
      cursor = el("g", {}, svg);
      cursor.line = el("line", { class: "playhead", y1: M.t, y2: M.t + ih }, cursor);
      cursor.dot = el("circle", { class: "dot", r: 4 }, cursor);
      cursor.value = text(cursor, 0, 0, "", "start", "value");
      svg.addEventListener("pointermove", (e) => { hover = timeAt(e); show(hover); });
      svg.addEventListener("pointerleave", () => { hover = null; show(now()); });
      svg.addEventListener("click", (e) => {
        const t = timeAt(e);
        if (video) video.currentTime = t;
        hover = null;
        show(t);
      });
      show(hover == null ? now() : hover);
    }

    function timeAt(e) {
      const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * (W / r.width);
      return Math.min(span(), Math.max(0, ((px - M.l) / (W - M.l - M.r)) * span()));
    }

    function show(t) {
      const cx = x(t), pt = latest(skill.points, t), flip = cx > W - 70;
      cursor.line.setAttribute("x1", cx);
      cursor.line.setAttribute("x2", cx);
      const has = pt && pt[1] != null;
      cursor.dot.setAttribute("visibility", has ? "visible" : "hidden");
      cursor.value.textContent = has ? pt[1].toFixed(2) : "";
      if (has) {
        cursor.dot.setAttribute("cx", cx);
        cursor.dot.setAttribute("cy", y(pt[1]));
        cursor.value.setAttribute("x", cx + (flip ? -8 : 8));
        cursor.value.setAttribute("y", Math.max(M.t + 12, y(pt[1]) - 8));
        cursor.value.setAttribute("text-anchor", flip ? "end" : "start");
      }
      if (readout) {
        let words = clock(t) + ": ";
        if (!pt) words += "before the first scored frame";
        else if (!has) words += "the student gave no score for this frame";
        else {
          words += "yes-probability " + pt[1].toFixed(2);
          if (th != null) {
            words += (pt[1] >= th ? ", over" : ", under") + " the threshold " + th.toFixed(2);
          }
        }
        readout.textContent = words;
      }
      if (banner && hover == null) {
        const a = skill.alerts.find((a) => t >= a.t && t < a.t + SHOW_ALERT_S);
        banner.hidden = !a;
        if (a) banner.textContent = "Spoken: " + a.text;
      }
    }

    draw();
    return { redraw: draw, update() { if (hover == null) show(now()); } };
  }

  window.Timeline = { mount };
})();
