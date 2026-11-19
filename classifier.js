const MODEL_URL = "https://teachablemachine.withgoogle.com/models/gVhWkIiqS/";
let model = null;
let batchResults = [];

async function loadModel() {
  if (model) return;
  setStatus("Cargando modelo desde TensorFlow.js...");
  model = await tmImage.load(MODEL_URL + "model.json", MODEL_URL + "metadata.json");
  setStatus("Modelo cargado correctamente.");
}

function setStatus(message) {
  document.getElementById("status").textContent = message;
}

function normalizeClassName(name) {
  if (!name) return "";
  const lower = name.toLowerCase();
  if (lower.includes("grafico")) return "Graficos";
  if (lower.includes("tabla")) return "Tablas";
  if (lower.includes("texto")) return "Textos";
  return "";
}

function inferTrueLabelFromPath(path) {
  if (!path) return "";
  const parts = path.split("/");
  for (let i = 0; i < parts.length - 1; i++) {
    const norm = normalizeClassName(parts[i]);
    if (norm) return norm;
  }
  return "";
}

function getProb(predictions, targetName) {
  const found = predictions.find(p => normalizeClassName(p.className) === targetName);
  return found ? found.probability : 0;
}

function buildProbCell(prob, barClass) {
  const pct = (prob * 100).toFixed(1);
  return `
    <div class="prob-cell">
      <div class="prob-text">${pct}%</div>
      <div class="mini-bar-bg">
        <div class="mini-bar-fill ${barClass}" style="width: ${pct}%;"></div>
      </div>
    </div>
  `;
}

function readFileAsImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => resolve({ img, src: e.target.result });
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function handleBatchUpload(event) {
  const files = Array.from(event.target.files || []).filter(f => f.type.startsWith("image/"));
  if (!files.length) return;

  await loadModel();
  const tbody = document.getElementById("results-body");
  tbody.innerHTML = "";
  batchResults = [];
  document.getElementById("evaluation-section").style.display = "none";

  setStatus(`Clasificando ${files.length} imagen(es)...`);
  let autoLabeledCount = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const relPath = file.webkitRelativePath || file.name;
    const inferredReal = inferTrueLabelFromPath(relPath);
    if (inferredReal) autoLabeledCount++;

    const { img, src } = await readFileAsImage(file);
    const predictions = await model.predict(img);

    predictions.sort((a, b) => b.probability - a.probability);
    const topClass = normalizeClassName(predictions[0].className) || predictions[0].className;

    const pGraficos = getProb(predictions, "Graficos");
    const pTablas = getProb(predictions, "Tablas");
    const pTextos = getProb(predictions, "Textos");

    batchResults.push({
      archivo: relPath,
      real: inferredReal,
      prediccion: topClass,
      probs: {
        Graficos: pGraficos,
        Tablas: pTablas,
        Textos: pTextos
      }
    });

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><img class="thumb" src="${src}" alt="${file.name}" /></td>
      <td>${relPath}</td>
      <td>
        <select class="true-label-select" data-index="${i}">
          <option value="" ${inferredReal === "" ? "selected" : ""}>-- Elegir clase real --</option>
          <option value="Graficos" ${inferredReal === "Graficos" ? "selected" : ""}>Graficos</option>
          <option value="Tablas" ${inferredReal === "Tablas" ? "selected" : ""}>Tablas</option>
          <option value="Textos" ${inferredReal === "Textos" ? "selected" : ""}>Textos</option>
        </select>
      </td>
      <td><span class="badge">${topClass} (${(predictions[0].probability * 100).toFixed(1)}%)</span></td>
      <td>${buildProbCell(pGraficos, "bar-graficos")}</td>
      <td>${buildProbCell(pTablas, "bar-tablas")}</td>
      <td>${buildProbCell(pTextos, "bar-textos")}</td>
    `;
    tbody.appendChild(tr);
  }

  if (autoLabeledCount === files.length) {
    setStatus(`Clasificación completada para ${files.length} imagen(es) con etiquetas detectadas automáticamente. Calculando métricas...`);
    runEvaluation();
  } else {
    setStatus(`Clasificación completada para ${files.length} imagen(es). Selecciona la "Clase Real" en cada fila y pulsa "Calcular Métricas y Curvas ROC".`);
  }

  event.target.value = "";
}

function runEvaluation() {
  const labeled = batchResults.filter(r => r.real !== "");
  if (!labeled.length) {
    alert("Primero clasifica imágenes y selecciona su 'Clase Real' en la tabla (o sube una carpeta con subcarpetas Graficos, Tablas y Textos).");
    return;
  }

  const report = EvaluationModule.evaluate(labeled);

  const metricsBody = document.getElementById("metrics-body");
  metricsBody.innerHTML = "";

  report.perClass.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${row.clase}</strong></td>
      <td>${(row.precision * 100).toFixed(1)}%</td>
      <td>${(row.recall * 100).toFixed(1)}%</td>
      <td>${(row.f1 * 100).toFixed(1)}%</td>
      <td><strong>${row.auc !== null ? row.auc.toFixed(3) : "N/A"}</strong></td>
    `;
    metricsBody.appendChild(tr);
  });

  const macroTr = document.createElement("tr");
  macroTr.style.backgroundColor = "#fef2f2";
  macroTr.innerHTML = `
    <td><strong>Promedio Macro</strong></td>
    <td><strong>${report.macroPrecision.toFixed(1)}%</strong></td>
    <td><strong>${report.macroRecall.toFixed(1)}%</strong></td>
    <td><strong>${report.macroF1.toFixed(1)}%</strong></td>
    <td><strong>${report.macroAuc !== null ? report.macroAuc.toFixed(3) : "N/A"}</strong></td>
  `;
  metricsBody.appendChild(macroTr);

  const cmBody = document.getElementById("cm-body");
  cmBody.innerHTML = "";
  EvaluationModule.CLASSES.forEach(realCls => {
    const tr = document.createElement("tr");
    const cells = EvaluationModule.CLASSES.map(predCls => {
      const val = report.matrix[realCls][predCls];
      let clsName = "cm-cell-zero";
      if (realCls === predCls && val > 0) clsName = "cm-cell-diag";
      else if (realCls !== predCls && val > 0) clsName = "cm-cell-err";
      return `<td class="${clsName}">${val}</td>`;
    }).join("");
    tr.innerHTML = `<td><strong>${realCls}</strong></td>${cells}`;
    cmBody.appendChild(tr);
  });

  document.getElementById("summary-pills").innerHTML = `
    <div class="summary-pill">Muestras evaluadas: ${report.total}</div>
    <div class="summary-pill">Exactitud (Accuracy): ${report.accuracy.toFixed(1)}%</div>
    <div class="summary-pill">F1-Score Macro: ${report.macroF1.toFixed(1)}%</div>
    <div class="summary-pill">AUC Macro: ${report.macroAuc !== null ? report.macroAuc.toFixed(3) : "N/A"}</div>
  `;

  EvaluationModule.drawROCCurves(document.getElementById("roc-canvas"), report.rocData);
  document.getElementById("evaluation-section").style.display = "block";
  setStatus(`Evaluación completada sobre ${report.total} imagen(es) etiquetadas.`);
}

function clearResults() {
  batchResults = [];
  document.getElementById("results-body").innerHTML = `
    <tr>
      <td colspan="7" style="text-align: center; color: #6b7280;">Resultados limpiados.</td>
    </tr>
  `;
  document.getElementById("evaluation-section").style.display = "none";
  setStatus("Listo para cargar nuevas imágenes.");
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("multi-upload").addEventListener("change", handleBatchUpload);
  document.getElementById("folder-upload").addEventListener("change", handleBatchUpload);
  document.getElementById("btn-evaluate").addEventListener("click", runEvaluation);
  document.getElementById("btn-clear").addEventListener("click", clearResults);

  document.getElementById("results-body").addEventListener("change", (event) => {
    if (event.target.classList.contains("true-label-select")) {
      const index = Number(event.target.dataset.index);
      if (batchResults[index]) {
        batchResults[index].real = event.target.value;
      }
    }
  });
});