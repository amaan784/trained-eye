// The training wall (signature visual 1): generated variations flip in one after
// another, each tagged with Muse Spark's label (green yes, red no); the ones Spark
// disagreed with, which are left out of training, fade out.
(function () {
  "use strict";

  function node(tag, cls, words) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (words != null) n.textContent = words;
    return n;
  }

  function describe(t) {
    const made = t.kind === "positive" ? "Made to show the condition" : "Made as a near-miss";
    let words = made + (t.condition ? ": " + t.condition : "") +
      (t.nuisance ? ", " + t.nuisance : "") + ".";
    if (!t.spark_label) return words + " Not labelled yet.";
    words += " Spark said " + t.spark_label + (t.kept ? "." : ", so it was rejected.");
    return t.reason ? words + " " + t.reason : words;
  }

  function tile(t, delayMs) {
    const fig = node("figure", "tile");
    fig.style.setProperty("--delay", delayMs + "ms");
    const src = t.image || t.url;
    if (t.error || !src) {
      fig.classList.add("failed");
      fig.textContent = t.error || "No image";
      return fig;
    }
    const img = node("img");
    img.src = src;
    img.loading = "lazy";
    img.alt = describe(t);
    fig.title = describe(t);
    fig.appendChild(img);
    if (t.spark_label) {
      const cls = t.spark_label === "yes" ? "yes" : t.spark_label === "no" ? "no" : "unsure";
      fig.appendChild(node("span", "tag " + cls, t.spark_label));
      if (!t.kept) {
        fig.classList.add("rejected");
        fig.appendChild(node("span", "dropped", "rejected"));
      }
    }
    return fig;
  }

  // The whole wall at once, staggered so the tiles flip in one by one.
  function render(box, tiles, stepMs) {
    const step = stepMs != null ? stepMs
      : Math.max(25, Math.min(90, 2400 / Math.max(1, tiles.length)));
    box.textContent = "";
    tiles.forEach((t, i) => box.appendChild(tile(t, Math.round(i * step))));
  }

  // One tile as it arrives (a live run).
  function add(box, t) { box.appendChild(tile(t, 0)); }

  function summary(tiles) {
    const labelled = tiles.filter((t) => t.spark_label);
    if (!labelled.length) return tiles.length ? "Not labelled by Spark yet." : "";
    const kept = labelled.filter((t) => t.kept).length;
    const said = (word) => labelled.filter((t) => t.spark_label === word).length;
    const yes = said("yes"), no = said("no"), unsure = labelled.length - yes - no;
    const parts = [`yes to ${yes}`, `no to ${no}`];
    if (unsure) parts.push(`unsure to ${unsure}`);
    return `Spark said ${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]} ` +
      `of these ${labelled.length}; ` +
      `${kept} agree with what they were made to show and are kept, ` +
      `${labelled.length - kept} are rejected.`;
  }

  window.Wall = { render, add, summary };
})();
