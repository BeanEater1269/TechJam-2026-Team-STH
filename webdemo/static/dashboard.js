// Transform categories from the challenge brief, grouped for small-multiple charts.
const TRANSFORM_GROUPS = [
  { title: "JPEG Compression", keys: ["jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30"], severityLabels: ["q90", "q70", "q50", "q30"] },
  { title: "Gaussian Blur", keys: ["blur_s0.5", "blur_s1.0", "blur_s2.0"], severityLabels: ["σ0.5", "σ1.0", "σ2.0"] },
  { title: "Resize", keys: ["resize_0.5x", "resize_0.25x"], severityLabels: ["0.5x", "0.25x"] },
  { title: "Gaussian Noise", keys: ["noise_s0.02", "noise_s0.05", "noise_s0.10"], severityLabels: ["σ0.02", "σ0.05", "σ0.10"] },
  { title: "Color Jitter", keys: ["color_jitter"], severityLabels: ["±20%"] },
  { title: "Center Crop", keys: ["crop_80pct"], severityLabels: ["80%"] },
];

const ALL_CONDITION_LABELS = {
  clean: "Clean",
  jpeg_q90: "JPEG q90", jpeg_q70: "JPEG q70", jpeg_q50: "JPEG q50", jpeg_q30: "JPEG q30",
  "blur_s0.5": "Blur σ0.5", "blur_s1.0": "Blur σ1.0", "blur_s2.0": "Blur σ2.0",
  "resize_0.5x": "Resize 0.5x", "resize_0.25x": "Resize 0.25x",
  "noise_s0.02": "Noise σ0.02", "noise_s0.05": "Noise σ0.05", "noise_s0.10": "Noise σ0.10",
  color_jitter: "Color Jitter", crop_80pct: "Crop 80%",
};

const MODEL_COLORS = ["#3ddc84", "#4da3ff"]; // model A, model B
const GRID_COLOR = "#23272f";
const TEXT_COLOR = "#7d8590";
const FONT_FAMILY = "JetBrains Mono, ui-monospace, monospace";

function renderCleanSummary(modelEntries) {
  const container = document.getElementById("clean-summary");
  container.innerHTML = modelEntries
    .map(
      ([modelKey, model], i) => `
      <div class="stat-card">
        <div class="stat-model" style="color:${MODEL_COLORS[i]}">${model.display_name}</div>
        <div class="stat-row"><span>Clean Accuracy</span><span class="mono">${(model.results.clean.accuracy * 100).toFixed(1)}%</span></div>
        <div class="stat-row"><span>Clean AUC</span><span class="mono">${model.results.clean.auc.toFixed(3)}</span></div>
      </div>`
    )
    .join("");
}

function renderGroupCharts(modelEntries) {
  const grid = document.getElementById("chart-grid");

  for (const group of TRANSFORM_GROUPS) {
    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `<h3>${group.title}</h3><div class="mini-chart-wrapper"><canvas></canvas></div>`;
    grid.appendChild(card);

    const canvas = card.querySelector("canvas");
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: group.severityLabels,
        datasets: modelEntries.map(([modelKey, model], i) => ({
          label: model.display_name,
          data: group.keys.map((key) => model.results[key]?.auc ?? null),
          backgroundColor: MODEL_COLORS[i],
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { grid: { color: GRID_COLOR }, ticks: { color: TEXT_COLOR, font: { family: FONT_FAMILY, size: 10 } } },
          y: {
            min: 0,
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
  }
}

function renderDetailTable(modelEntries) {
  const tbody = document.querySelector("#detail-table tbody");
  const rows = Object.keys(ALL_CONDITION_LABELS)
    .map((key) => {
      const cells = modelEntries
        .map(([, model]) => {
          const r = model.results[key];
          return `<td>${(r.accuracy * 100).toFixed(1)}%</td><td>${r.auc.toFixed(3)}</td>`;
        })
        .join("");
      return `<tr><td>${ALL_CONDITION_LABELS[key]}</td>${cells}</tr>`;
    })
    .join("");
  tbody.innerHTML = rows;
}

async function main() {
  const response = await fetch("/static/results.json", { cache: "no-store" });
  const data = await response.json();
  const modelEntries = Object.entries(data.models);

  renderCleanSummary(modelEntries);
  renderGroupCharts(modelEntries);
  renderDetailTable(modelEntries);
}

main();
