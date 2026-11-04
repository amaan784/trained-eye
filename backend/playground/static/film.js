// Film for the playground (DESIGN.md): strips of real frames with sprocket holes and edge
// print, SAM's boxes on teaching frames, and the grease-pencil ring that draws itself
// once, around the frame where the real alert fired.
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function node(tag, cls, words) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (words != null) n.textContent = words;
    return n;
  }

  // 0:16, or 0:05.7 when frames share a second
  function time(t, tenths) {
    const m = Math.floor(t / 60), s = t - m * 60;
    return m + ":" + (tenths ? s.toFixed(1).padStart(4, "0") : String(Math.floor(s)).padStart(2, "0"));
  }

  // A strip: cells [{src | canvas, edge, size: [w, h], box: [x1, y1, x2, y2], alt}]
  function strip(cells, opts) {
    opts = opts || {};
    const film = node("div", "film" + (opts.bleed === false ? "" : " bleed"));
    const track = node("div", "track");
    const top = node("div", "holes top"), bottom = node("div", "holes bottom");
    track.append(top, bottom);
    cells.forEach((c) => track.appendChild(cell(c)));
    film.appendChild(track);
    requestAnimationFrame(() => holes(track));
    return { film, track };
  }

  function holes(track) {
    const width = track.scrollWidth;
    for (const row of track.querySelectorAll(".holes")) {
      row.textContent = "";
      for (let x = 6; x < width - 8; x += 16) {
        const h = node("span", "hole");
        h.style.left = x + "px";
        row.appendChild(h);
      }
    }
  }

  function cell(c) {
    const box = node(c.input ? "label" : "div", "cell");
    if (c.input) box.appendChild(c.input);
    const frame = node("div", "frame");
    if (c.canvas) frame.appendChild(c.canvas);
    else if (c.src) {
      const img = node("img");
      img.src = c.src;
      img.alt = c.alt || "";
      img.loading = "lazy";
      frame.appendChild(img);
    }
    if (c.box && c.size) frame.appendChild(samBox(c.box, c.size));
    box.append(frame, node("span", "edge", c.edge || ""));
    if (c.title) box.title = c.title;
    return box;
  }

  // SAM's box, placed as object-fit: cover places the frame in a 3:4 cell
  function samBox(b, size) {
    const span = node("span", "sam");
    const [w, h] = size, s = Math.max(3 / w, 4 / h);
    const ox = (3 - w * s) / 2, oy = (4 - h * s) / 2;
    const pct = (v, total) => (v / total * 100).toFixed(2) + "%";
    span.style.left = pct(ox + b[0] * s, 3);
    span.style.top = pct(oy + b[1] * s, 4);
    span.style.width = pct((b[2] - b[0]) * s, 3);
    span.style.height = pct((b[3] - b[1]) * s, 4);
    return span;
  }

  // The design kit's hand-drawn ellipse: it wobbles, opens up as it comes round again
  // and overshoots its start. A small seeded generator keeps it the same every time.
  function seeded(seed) {
    let a = seed >>> 0;
    return () => { a = (a * 1664525 + 1013904223) >>> 0; return a / 4294967296; };
  }
  function handEllipse(cx, cy, rx, ry, rotDeg, seed, overshoot) {
    const rnd = seeded(seed), rot = rotDeg * Math.PI / 180;
    const start = (195 + rnd() * 20) * Math.PI / 180;
    const sweep = 2 * Math.PI * (1 + overshoot);
    const wobble = Array.from({ length: 8 }, () => rnd() * 2 - 1);
    const pts = [];
    for (let i = 0; i <= 90; i++) {
      const a = start + sweep * i / 90;
      const noise = 1 + 0.035 * wobble.reduce((sum, w, k) => sum + w * Math.sin((k + 1) * a + k), 0) / 4;
      const drift = 1 + 0.06 * i / 90;
      const x0 = rx * noise * drift * Math.cos(a), y0 = ry * noise * drift * Math.sin(a);
      pts.push([cx + x0 * Math.cos(rot) - y0 * Math.sin(rot), cy + x0 * Math.sin(rot) + y0 * Math.cos(rot)]);
    }
    return "M " + pts.map((p) => p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" L ");
  }

  // Ring the cell (inside its track, so it travels with the film); draws once in 600 ms
  function ring(track, cellEl) {
    clearRing(track);
    const w = cellEl.offsetWidth, h = cellEl.offsetHeight;
    const cx = cellEl.offsetLeft + w / 2, cy = cellEl.offsetTop + h / 2;
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("class", "marks");
    svg.setAttribute("width", track.scrollWidth);
    svg.setAttribute("height", track.offsetHeight);
    svg.style.top = "0";
    svg.innerHTML = '<defs><filter id="wax" x="-10%" y="-10%" width="120%" height="120%">' +
      '<feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="4" result="n"/>' +
      '<feDisplacementMap in="SourceGraphic" in2="n" scale="2.2"/></filter></defs>';
    const g = document.createElementNS(NS, "g");
    g.setAttribute("filter", "url(#wax)");
    const passes = [[handEllipse(cx, cy, w * 0.68, h * 0.62, -7, 3, 0.18), 5.5, 0.95],
                    [handEllipse(cx + 2, cy - 1, w * 0.70, h * 0.63, -5, 11, 0.08), 2.6, 0.7]];
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    passes.forEach(([d, width, opacity], i) => {
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", d);
      p.setAttribute("stroke-width", width);
      p.setAttribute("opacity", opacity);
      g.appendChild(p);
      if (!still) {
        p.style.visibility = "hidden"; // until its length is known in the document
        requestAnimationFrame(() => {
          const total = p.getTotalLength();
          p.style.strokeDasharray = total;
          p.style.strokeDashoffset = total;
          p.style.visibility = "";
          p.animate([{ strokeDashoffset: total }, { strokeDashoffset: 0 }],
            { duration: 600, delay: i * 120, easing: "cubic-bezier(.3,.6,.4,1)", fill: "forwards" });
        });
      }
    });
    svg.appendChild(g);
    track.appendChild(svg);
  }
  function clearRing(track) {
    for (const old of track.querySelectorAll("svg.marks")) old.remove();
  }

  window.Film = { strip, cell, time, ring, clearRing, holes, node };
})();
