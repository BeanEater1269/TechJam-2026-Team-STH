// Transform categories from the challenge brief, grouped for small-multiple charts.
// Keys match the real variant names produced by the team's build_cache.py pipeline.
const TRANSFORM_GROUPS = [
  { title: "JPEG Compression", keys: ["jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30"], severityLabels: ["q90", "q70", "q50", "q30"] },
  { title: "Gaussian Blur", keys: ["blur_s0.5", "blur_s1.0", "blur_s2.0"], severityLabels: ["σ0.5", "σ1.0", "σ2.0"] },
  { title: "Resize", keys: ["resize_0.5", "resize_0.25"], severityLabels: ["0.5x", "0.25x"] },
  { title: "Gaussian Noise", keys: ["noise_s0.02", "noise_s0.05", "noise_s0.1"], severityLabels: ["σ0.02", "σ0.05", "σ0.1"] },
  { title: "Color Jitter", keys: ["color_down", "color_up"], severityLabels: ["darker", "brighter"] },
  { title: "Center Crop", keys: ["crop80"], severityLabels: ["80%"] },
];

const ALL_CONDITION_LABELS = {
  clean: "Clean",
  jpeg_q90: "JPEG q90", jpeg_q70: "JPEG q70", jpeg_q50: "JPEG q50", jpeg_q30: "JPEG q30",
  "blur_s0.5": "Blur σ0.5", "blur_s1.0": "Blur σ1.0", "blur_s2.0": "Blur σ2.0",
  "resize_0.5": "Resize 0.5x", "resize_0.25": "Resize 0.25x",
  "noise_s0.02": "Noise σ0.02", "noise_s0.05": "Noise σ0.05", "noise_s0.1": "Noise σ0.1",
  color_down: "Color darker", color_up: "Color brighter", crop80: "Crop 80%",
};

const MODEL_ORDER = ["base", "concat", "film"];
const MODEL_COLORS = { base: "#7d8590", concat: "#3ddc84", film: "#4da3ff" }; // base=grey control, concat=green, film=blue
const GRID_COLOR = "#23272f";
const TEXT_COLOR = "#7d8590";
const FONT_FAMILY = "JetBrains Mono, ui-monospace, monospace";

let DATA = null;
let charts = [];

function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

function renderSummary(backboneKey) {
  const models = DATA.backbones[backboneKey].models;
  const container = document.getElementById("clean-summary");
  container.innerHTML = MODEL_ORDER.map((modelKey) => {
    const m = models[modelKey];
    return `
      <div class="stat-card">
        <div class="stat-model" style="color:${MODEL_COLORS[modelKey]}">${m.display_name}</div>
        <div class="stat-row"><span>Overall Accuracy</span><span class="mono">${pct(m.summary.overall.accuracy)}</span></div>
        <div class="stat-row"><span>Overall AUC</span><span class="mono">${m.summary.overall.auc.toFixed(4)}</span></div>
        <div class="stat-row"><span>Clean Accuracy</span><span class="mono">${pct(m.summary.clean.acc)}</span></div>
        <div class="stat-row"><span>Robustness Gap (acc)</span><span class="mono">${(m.summary.gap.acc * 100).toFixed(2)} pts</span></div>
      </div>`;
  }).join("");
}

function renderGroupCharts(backboneKey) {
  const models = DATA.backbones[backboneKey].models;
  const grid = document.getElementById("chart-grid");
  grid.innerHTML = "";
  charts.forEach((c) => c.destroy());
  charts = [];

  for (const group of TRANSFORM_GROUPS) {
    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `<h3>${group.title}</h3><div class="mini-chart-wrapper"><canvas></canvas></div>`;
    grid.appendChild(card);

    const canvas = card.querySelector("canvas");
    const chart = new Chart(canvas, {
      type: "bar",
      data: {
        labels: group.severityLabels,
        datasets: MODEL_ORDER.map((modelKey) => ({
          label: models[modelKey].display_name,
          data: group.keys.map((key) => models[modelKey].results[key]?.auc ?? null),
          backgroundColor: MODEL_COLORS[modelKey],
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: GRID_COLOR }, ticks: { color: TEXT_COLOR, font: { family: FONT_FAMILY, size: 10 } } },
          y: {
            min: 0.8,
            max: 1,
            grid: { color: GRID_COLOR },
            ticks: { color: TEXT_COLOR, font: { family: FONT_FAMILY, size: 10 } },
          },
        },
        plugins: {
          legend: { display: false },
        },
      },
    });
    charts.push(chart);
  }
}

function renderDetailTable(backboneKey) {
  const models = DATA.backbones[backboneKey].models;
  const tbody = document.querySelector("#detail-table tbody");
  const rows = Object.keys(ALL_CONDITION_LABELS)
    .map((key) => {
      const cells = MODEL_ORDER.map((modelKey) => {
        const r = models[modelKey].results[key];
        return `<td>${pct(r.accuracy)}</td><td>${r.auc.toFixed(3)}</td>`;
      }).join("");
      return `<tr><td>${ALL_CONDITION_LABELS[key]}</td>${cells}</tr>`;
    })
    .join("");
  tbody.innerHTML = rows;
}

function renderAll(backboneKey) {
  renderSummary(backboneKey);
  renderGroupCharts(backboneKey);
  renderDetailTable(backboneKey);

  document.querySelectorAll(".backbone-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.backbone === backboneKey);
  });
}

async function main() {
  const response = await fetch("/static/results.json", { cache: "no-store" });
  DATA = await response.json();

  document.querySelectorAll(".backbone-tab").forEach((btn) => {
    btn.addEventListener("click", () => renderAll(btn.dataset.backbone));
  });

  renderAll("l14");
}

main();
