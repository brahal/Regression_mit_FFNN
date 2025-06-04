// --- Hilfsfunktionen ---
function trueFunction(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function gaussianNoise(stdDev = 0.2236) {
    let u = Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * stdDev;
}

function generateData(n = 100, noise = false) {
    const data = [];
    for (let i = 0; i < n; i++) {
        const x = Math.random() * 4 - 2;
        const y = trueFunction(x) + (noise ? gaussianNoise(0.2236) : 0);
        data.push({ x, y });
    }
    return data.sort((a, b) => a.x - b.x);
}

function splitData(data) {
    const shuffled = [...data].sort(() => 0.5 - Math.random());
    const train = shuffled.slice(0, data.length / 2);
    const test = shuffled.slice(data.length / 2);
    return { train, test };
}

function toTensors(data) {
    const xs = tf.tensor2d(data.map(d => [d.x]));
    const ys = tf.tensor2d(data.map(d => [d.y]));
    return { xs, ys };
}

// --- Spinner ---
function showSpinner(id) {
    const spinner = document.querySelector(`.spinner[data-target="${id}"]`);
    if (spinner) spinner.style.display = "block";
}

function hideSpinner(id) {
    const spinner = document.querySelector(`.spinner[data-target="${id}"]`);
    if (spinner) spinner.style.display = "none";
}

// --- Diagramm: Trainings- und Testdaten ---
function plotScatter(id, title, train, test) {
    showSpinner(id);

    // Verzögert Plotly.newPlot minimal → Browser kann vorher Spinner anzeigen
    setTimeout(() => {
        Plotly.newPlot(id, [
            {
                x: train.map(d => d.x),
                y: train.map(d => d.y),
                mode: 'markers',
                name: 'Trainingsdaten',
                marker: { color: '#1f5aa6', size: 8, opacity: 0.9, symbol: 'circle', line: { width: 1, color: '#ffffff' }},
                type: 'scatter'
            },
            {
                x: test.map(d => d.x),
                y: test.map(d => d.y),
                mode: 'markers',
                name: 'Testdaten',
                marker: { color: '#e6550d', size: 8, opacity: 0.9, symbol:'x', line: { width: 1, color: '#ffffff' } },
                type: 'scatter'
            }
        ], {
            xaxis: {
                title: 'x',
                showline: true,
                linecolor: '#aaa',
                linewidth: 1,
                mirror: false,
                ticks: 'outside',
                tickcolor: '#ccc',
                ticklen: 6,
                zeroline: false,
                gridcolor: '#f5f5f5'
            },
            yaxis: {
                title: 'y',
                showline: true,
                linecolor: '#aaa',
                linewidth: 1,
                mirror: false,
                ticks: 'outside',
                tickcolor: '#ccc',
                ticklen: 6,
                zeroline: false,
                gridcolor: '#f5f5f5'
            },
            legend: { orientation: 'h', xanchor: 'center', x: 0.5, y: -0.2 },
            margin: { t: 20 },
            font: {
                family: 'Segoe UI, sans-serif',
                size: 14,
                color: '#333'
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        }).then(() => {
            hideSpinner(id);
            const titleEl = document.getElementById(id + "-title");
            if (titleEl) titleEl.textContent = title;
        });
    }, 50); // 💡 kleine Verzögerung reicht
}


// --- Modelltraining ---
async function trainModel(dataTrain, epochs = 100) {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1], units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

    const { xs, ys } = toTensors(dataTrain);
    await model.fit(xs, ys, { epochs, batchSize: 32, shuffle: true });
    return model;
}

// --- Vorhersagen plotten & MSE berechnen ---
async function evaluateAndPlot(model, id, data, label) {
    showSpinner(id); // Spinner sofort aktivieren

    const xs = tf.tensor2d(data.map(d => [d.x]));
    const preds = model.predict(xs);
    const predYs = await preds.array();

    // Rendering verzögern, damit Spinner sichtbar bleibt
    setTimeout(() => {
        Plotly.newPlot(id, [
            {
                x: data.map(d => d.x),
                y: data.map(d => d.y),
                mode: 'markers',
                name: 'Reale Werte',
                marker: { color: '#000000', size: 6, opacity: 0.9 },
                type: 'scatter'
            },
            {
                x: data.map(d => d.x),
                y: predYs.map(p => p[0]),
                mode: 'lines',
                name: 'Vorhersage',
                line: { color: '#32cd32', width: 2 }
            }
        ], {
            xaxis: {
                title: 'x',
                showline: true,
                linecolor: '#aaa',
                linewidth: 1,
                mirror: false,
                ticks: 'outside',
                tickcolor: '#ccc',
                ticklen: 6,
                zeroline: false,
                gridcolor: '#f5f5f5'
            },
            yaxis: {
                title: 'y',
                showline: true,
                linecolor: '#aaa',
                linewidth: 1,
                mirror: false,
                ticks: 'outside',
                tickcolor: '#ccc',
                ticklen: 6,
                zeroline: false,
                gridcolor: '#f5f5f5'
            },
            legend: {
                orientation: 'h',
                xanchor: 'center',
                x: 0.5,
                y: -0.2
            },
            margin: { t: 20 },
            font: {
                family: 'Segoe UI, sans-serif',
                size: 14,
                color: '#333'
            },
            plot_bgcolor: '#ffffff',
            paper_bgcolor: '#ffffff'
        }).then(() => {
            hideSpinner(id); // Spinner nach dem Plot entfernen
        });

        const titleEl = document.getElementById(id + "-title");
        if (titleEl) titleEl.textContent = `${label} – Vorhersage vs. Reale Werte`;
    }, 50);

    const ys = tf.tensor2d(data.map(d => [d.y]));
    const mse = tf.losses.meanSquaredError(ys, preds).dataSync()[0];
    return mse.toFixed(4);
}

// --- Alles starten ---
async function runAll() {
    // R1 sofort anzeigen
    const cleanData = generateData(100, false);
    const noisyData = generateData(100, true);
    const { train: cleanTrain, test: cleanTest } = splitData(cleanData);
    const { train: noisyTrain, test: noisyTest } = splitData(noisyData);
    plotScatter("plot-data-clean", "Unverrauschte Daten", cleanTrain, cleanTest);
    plotScatter("plot-data-noisy", "Verrauschte Daten", noisyTrain, noisyTest);

    // R2 – Clean Modell
    showSpinner("prediction-clean-train");
    showSpinner("prediction-clean-test");
    await new Promise(resolve => setTimeout(resolve, 100));
    const modelClean = await trainModel(cleanTrain, 100);
    const mseCleanTrain = await evaluateAndPlot(modelClean, "prediction-clean-train", cleanTrain, "Train Clean");
    const mseCleanTest = await evaluateAndPlot(modelClean, "prediction-clean-test", cleanTest, "Test Clean");
    document.getElementById("loss-clean").innerText = `Loss Train: ${mseCleanTrain}, Test: ${mseCleanTest}`;

    // R3 – Best-Fit
    showSpinner("prediction-best-train");
    showSpinner("prediction-best-test");
    await new Promise(resolve => setTimeout(resolve, 100));
    const modelBest = await trainModel(noisyTrain, 100);
    const mseBestTrain = await evaluateAndPlot(modelBest, "prediction-best-train", noisyTrain, "Train Best");
    const mseBestTest = await evaluateAndPlot(modelBest, "prediction-best-test", noisyTest, "Test Best");
    document.getElementById("loss-best").innerText = `Loss Train: ${mseBestTrain}, Test: ${mseBestTest}`;

    // R4 – Overfit
    showSpinner("prediction-overfit-train");
    showSpinner("prediction-overfit-test");
    await new Promise(resolve => setTimeout(resolve, 100));
    const modelOverfit = await trainModel(noisyTrain, 500);
    const mseOverfitTrain = await evaluateAndPlot(modelOverfit, "prediction-overfit-train", noisyTrain, "Train Overfit");
    const mseOverfitTest = await evaluateAndPlot(modelOverfit, "prediction-overfit-test", noisyTest, "Test Overfit");
    document.getElementById("loss-overfit").innerText = `Loss Train: ${mseOverfitTrain}, Test: ${mseOverfitTest}`;
}

runAll();