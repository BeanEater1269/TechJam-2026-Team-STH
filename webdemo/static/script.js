const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const previewImage = document.getElementById("preview-image");
const resultLabel = document.getElementById("result-label");
const resultScore = document.getElementById("result-score");
const resultRaw = document.getElementById("result-raw");
const tryAgainButton = document.getElementById("try-again");

function showOnly(sectionToShow) {
  for (const section of [dropzone, loading, result]) {
    section.classList.toggle("hidden", section !== sectionToShow);
  }
}

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Please choose an image file.");
    return;
  }

  previewImage.src = URL.createObjectURL(file);
  showOnly(loading);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "prediction failed");
    showResult(data.pred);
  } catch (err) {
    alert("Something went wrong talking to the server. Check the console.");
    console.error(err);
    showOnly(dropzone);
  }
}

function showResult(pred) {
  const isLikelyAi = pred >= 0.5;
  // Confidence in the ASSIGNED label, not the raw score -- pred=0.02 predicting
  // REAL should read "98% confidence", not "2% confidence".
  const confidence = isLikelyAi ? pred : 1 - pred;

  resultLabel.textContent = isLikelyAi ? "LIKELY AI-GENERATED" : "LIKELY REAL";
  resultLabel.className = isLikelyAi ? "fake" : "real";
  resultScore.textContent = `${Math.round(confidence * 100)}% confidence`;
  resultRaw.textContent = `p(AI-generated) = ${pred.toFixed(4)}`;

  showOnly(result);
}

// Click the dropzone to open the normal file picker
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

// Drag-and-drop support
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
  handleFile(e.dataTransfer.files[0]);
});

tryAgainButton.addEventListener("click", () => {
  fileInput.value = "";
  showOnly(dropzone);
});
