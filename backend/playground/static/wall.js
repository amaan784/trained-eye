// Generated variations as a contact sheet: each frame on film with Muse Spark's yes or no
// in edge print under it. The ones Spark disagreed with, which are left out of training,
// are dimmed.
(function () {
  "use strict";

  function describe(t) {
    const made = t.kind === "positive" ? "Made to show the condition" : "Made as a near-miss";
    let words = made + (t.condition ? ": " + t.condition : "") +
      (t.nuisance ? ", " + t.nuisance : "") + ".";
    if (!t.spark_label) return words + " Not labelled yet.";
    words += " Spark said " + t.spark_label + (t.kept ? "." : ", so it was left out.");
    return t.reason ? words + " " + t.reason : words;
  }

  function cellOf(t) {
    const src = t.image || t.url;
    if (t.error || !src) return Film.cell({ edge: "", title: t.error || "No image" });
    const c = Film.cell({ src, edge: t.spark_label || "", alt: describe(t), title: describe(t) });
    if (t.spark_label && !t.kept) c.classList.add("dim");
    return c;
  }

  function track(box) {
    const found = box.querySelector(".track");
    if (found) return found;
    const s = Film.strip([], {});
    box.appendChild(s.film);
    return s.track;
  }

  function render(box, tiles) {
    box.textContent = "";
    const t = track(box);
    tiles.forEach((tile) => t.appendChild(cellOf(tile)));
    requestAnimationFrame(() => Film.holes(t));
  }

  function add(box, tile) {
    const t = track(box);
    t.appendChild(cellOf(tile));
    requestAnimationFrame(() => Film.holes(t));
  }

  function summary(tiles) {
    const labelled = tiles.filter((t) => t.spark_label);
    if (!labelled.length) return tiles.length ? "Not labelled by Spark yet." : "";
    const kept = labelled.filter((t) => t.kept).length;
    const said = (word) => labelled.filter((t) => t.spark_label === word).length;
    const yes = said("yes"), no = said("no"), unsure = labelled.length - yes - no;
    const parts = [`yes to ${yes}`, `no to ${no}`];
    if (unsure) parts.push(`unsure to ${unsure}`);
    return `Spark said ${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]} ` +
      `of these ${labelled.length}; ${kept} agree with what they were made to show and ` +
      `are kept, ${labelled.length - kept} are left out.`;
  }

  window.Wall = { render, add, summary };
})();
