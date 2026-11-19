const EvaluationModule = (() => {
  const CLASSES = ["Graficos", "Tablas", "Textos"];

  function computeROCAndAUC(yTrueBinary, yScores) {
    const pairs = yTrueBinary.map((y, idx) => ({ y, score: yScores[idx] }));
    pairs.sort((a, b) => b.score - a.score);

    const P = pairs.filter(p => p.y === 1).length;
    const N = pairs.filter(p => p.y === 0).length;

    if (P === 0 || N === 0) {
      return { points: [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }], auc: null };
    }

    let tp = 0;
    let fp = 0;
    const points = [{ fpr: 0, tpr: 0 }];

    for (let i = 0; i < pairs.length; i++) {
      if (pairs[i].y === 1) tp++;
      else fp++;
      points.push({ fpr: fp / N, tpr: tp / P });
    }

    let auc = 0;
    for (let i = 1; i < points.length; i++) {
      const dx = points[i].fpr - points[i - 1].fpr;
      const avgY = (points[i].tpr + points[i - 1].tpr) / 2;
      auc += dx * avgY;
    }

    return { points, auc };
  }

  function drawROCCurves(canvas, rocData) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const padL = 52;
    const padR = 20;
    const padT = 24;
    const padB = 48;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);

    // Cuadrícula de referencia
    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 5; g++) {
      const x = padL + (plotW * g) / 5;
      const y = padT + (plotH * g) / 5;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
    }

    // Ejes principales
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    // Diagonal aleatoria (AUC = 0.5)
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "#9ca3af";
    ctx.beginPath();
    ctx.moveTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT);
    ctx.stroke();
    ctx.restore();

    // Etiquetas de ejes
    ctx.fillStyle = "#374151";
    ctx.font = "12px Segoe UI, Arial";
    ctx.textAlign = "center";
    for (let g = 0; g <= 5; g++) {
      const val = (g / 5).toFixed(1);
      const x = padL + (plotW * g) / 5;
      ctx.fillText(val, x, padT + plotH + 18);
    }
    ctx.fillText("Tasa de Falsos Positivos (FPR)", padL + plotW / 2, H - 10);

    ctx.textAlign = "right";
    for (let g = 0; g <= 5; g++) {
      const val = (1 - g / 5).toFixed(1);
      const y = padT + (plotH * g) / 5 + 4;
      ctx.fillText(val, padL - 8, y);
    }

    ctx.save();
    ctx.translate(15, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Tasa de Verdaderos Positivos (TPR)", 0, 0);
    ctx.restore();

    // Curvas por clase
    const colors = {
      Graficos: "#ea580c",
      Tablas: "#db2777",
      Textos: "#7c3aed"
    };

    const legendY = padT + plotH - 68;
    const legendX = padL + plotW - 185;

    CLASSES.forEach((cls, idx) => {
      const item = rocData[cls];
      ctx.strokeStyle = colors[cls];
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      item.points.forEach((pt, pIdx) => {
        const cx = padL + pt.fpr * plotW;
        const cy = padT + (1 - pt.tpr) * plotH;
        if (pIdx === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      });
      ctx.stroke();

      const ly = legendY + idx * 20;
      ctx.fillStyle = colors[cls];
      ctx.fillRect(legendX, ly - 9, 14, 4);
      ctx.fillStyle = "#1f2937";
      ctx.font = "bold 12px Segoe UI, Arial";
      ctx.textAlign = "left";
      const aucText = item.auc !== null ? item.auc.toFixed(3) : "N/A";
      ctx.fillText(`${cls} (AUC = ${aucText})`, legendX + 20, ly - 3);
    });
  }

  function evaluate(labeledResults) {
    const matrix = {
      Graficos: { Graficos: 0, Tablas: 0, Textos: 0 },
      Tablas: { Graficos: 0, Tablas: 0, Textos: 0 },
      Textos: { Graficos: 0, Tablas: 0, Textos: 0 }
    };

    let correct = 0;
    labeledResults.forEach(item => {
      if (matrix[item.real] && matrix[item.real][item.prediccion] !== undefined) {
        matrix[item.real][item.prediccion]++;
      }
      if (item.real === item.prediccion) correct++;
    });

    const perClass = [];
    const rocData = {};
    let sumPrec = 0, sumRec = 0, sumF1 = 0, sumAuc = 0, validAucCount = 0;

    CLASSES.forEach(cls => {
      const tp = matrix[cls][cls];
      const fp = CLASSES.reduce((acc, r) => acc + (r !== cls ? matrix[r][cls] : 0), 0);
      const fn = CLASSES.reduce((acc, c) => acc + (c !== cls ? matrix[cls][c] : 0), 0);

      const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
      const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
      const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

      const yTrueBin = labeledResults.map(item => (item.real === cls ? 1 : 0));
      const yScores = labeledResults.map(item => item.probs[cls]);
      const roc = computeROCAndAUC(yTrueBin, yScores);
      rocData[cls] = roc;

      sumPrec += precision;
      sumRec += recall;
      sumF1 += f1;
      if (roc.auc !== null) {
        sumAuc += roc.auc;
        validAucCount++;
      }

      perClass.push({
        clase: cls,
        precision,
        recall,
        f1,
        auc: roc.auc
      });
    });

    return {
      total: labeledResults.length,
      accuracy: (correct / labeledResults.length) * 100,
      macroPrecision: (sumPrec / CLASSES.length) * 100,
      macroRecall: (sumRec / CLASSES.length) * 100,
      macroF1: (sumF1 / CLASSES.length) * 100,
      macroAuc: validAucCount > 0 ? sumAuc / validAucCount : null,
      perClass,
      matrix,
      rocData
    };
  }

  return {
    CLASSES,
    evaluate,
    drawROCCurves
  };
})();