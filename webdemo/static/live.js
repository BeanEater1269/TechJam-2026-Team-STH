// Standalone -- does NOT import from results-dashboard.js, whose render functions are
// hardcoded to the eval-results JSON shape (backbones/models/per_variant), not this
// page's judge-run shape.

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const loading = document.getElementById("loading");
const loadingText = document.getElementById("loading-text");
const singleResult = document.getElementById("single-result");
const batchResult = document.getElementById("batch-result");
const previewImage = document.getElementById("preview-image");
const resultLabel = document.getElementById("result-label");
const resultScore = document.getElementById("result-score");
const resultRaw = document.getElementById("result-raw");
const batchRunId = document.getElementById("batch-run-id");
const tryAgainButton = document.getElementById("try-again");
const liveSummary = document.getElementById("live-summary");
const liveTableWrap = document.getElementById("live-table-wrap");
const liveTableBody = document.querySelector("#live-table tbody");

const MAX_FILES = 50;

function showOnly(sectionToShow) {
  for (const section of [dropzone, loading, singleResult, batchResult]) {
    section.classList.toggle("hidden", section !== sectionToShow);
  }
  tryAgainButton.classList.toggle("hidden", sectionToShow === dropzone || sectionToShow === loading);
  if (sectionToShow !== batchResult) {
    liveSummary.innerHTML = "";
    liveTableWrap.classList.add("hidden");
  }
}

async function handleFiles(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  if (files.length === 0) {
    alert("Please choose image file(s).");
    return;
  }
  if (files.length > MAX_FILES) {
    alert(`Please choose at most ${MAX_FILES} images.`);
    return;
  }

  loadingText.textContent = files.length === 1 ? "Analyzing..." : `Analyzing ${files.length} images...`;
  showOnly(loading);

  const formData = new FormData();
  for (const f of files) formData.append("files", f);

  try {
    const response = await fetch("/api/live/analyze", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "analysis failed");

    if (data.mode === "single") {
      // Single image: inline result only. Nothing was persisted server-side, and this
      // branch never touches #live-summary/#live-table-wrap.
      renderSingle(data.result, URL.createObjectURL(files[0]));
    } else {
      // Batch: the server already saved the run to disk before responding with just a
      // pointer -- pull it back and render FROM that saved copy, same "disk is the
      // source of truth" pattern as results_b32/results_l14.
      const runResp = await fetch(data.run_url);
      const run = await runResp.json();
      renderBatch(run);
    }
  } catch (err) {
    alert("Something went wrong talking to the server. Check the console.");
    console.error(err);
    showOnly(dropzone);
  }
}

function renderSingle(result, previewUrl) {
  previewImage.src = previewUrl;
  const isLikelyAi = result.label === "FAKE";

  resultLabel.textContent = isLikelyAi ? "LIKELY AI-GENERATED" : "LIKELY REAL";
  resultLabel.className = isLikelyAi ? "fake" : "real";
  resultScore.textContent = `${Math.round(result.confidence * 100)}% confidence`;
  resultRaw.textContent = `p(AI-generated) = ${result.raw_prob.toFixed(4)}`;

  showOnly(singleResult);
}

function renderBatch(run) {
  batchRunId.textContent = `Run ${run.run_id} — ${run.n_images} image(s) analyzed`;
  showOnly(batchResult);
  renderLiveSummary(run);
  renderLiveTable(run);
}

function renderLiveSummary(run) {
  const s = run.results.summary;
  const card = document.createElement("div");
  card.className = "stat-card";
  card.innerHTML = `
    <div class="stat-model">Batch summary</div>
    <div class="stat-row"><span>Images analyzed</span><span class="mono">${s.n}</span></div>
    <div class="stat-row"><span>Flagged AI-generated</span><span class="mono">${s.n_fake} (${Math.round(s.fake_rate * 100)}%)</span></div>
    <div class="stat-row"><span>Mean confidence</span><span class="mono">${Math.round(s.mean_confidence * 100)}%</span></div>
    <div class="stat-row"><span>Lowest confidence</span><span class="mono">${s.min_confidence !== null ? Math.round(s.min_confidence * 100) + "%" : "--"}</span></div>
  `;
  liveSummary.innerHTML = "";
  liveSummary.appendChild(card);
}

function renderLiveTable(run) {
  liveTableBody.innerHTML = "";
  for (const img of run.results.per_image) {
    const isLikelyAi = img.label === "FAKE";
    const thumbUrl = `/judge-files/${run.run_id}/${img.stored_image}`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><img class="thumb" src="${thumbUrl}" alt=""></td>
      <td>${img.original_filename}</td>
      <td><span class="badge ${isLikelyAi ? "fake" : "real"}">${isLikelyAi ? "AI-GENERATED" : "REAL"}</span></td>
      <td class="mono">${Math.round(img.confidence * 100)}%</td>
      <td class="mono">${img.raw_prob.toFixed(4)}</td>
    `;
    liveTableBody.appendChild(tr);
  }
  liveTableWrap.classList.remove("hidden");
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFiles(fileInput.files));

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  handleFiles(e.dataTransfer.files);
});

tryAgainButton.addEventListener("click", () => {
  fileInput.value = "";
  showOnly(dropzone);
});
