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

function plotScatter(id, title, train, test) {
    Plotly.newPlot(id, [
        {
            x: train.map(d => d.x),
            y: train.map(d => d.y),
            mode: 'markers',
            name: 'Train',
            marker: { color: 'blue' }
        },
        {
            x: test.map(d => d.x),
            y: test.map(d => d.y),
            mode: 'markers',
            name: 'Test',
            marker: { color: 'red' }
        }
    ], {
        title,
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    });
}

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

async function evaluateAndPlot(model, id, data, label) {
    const xs = tf.tensor2d(data.map(d => [d.x]));
    const preds = model.predict(xs);
    const predYs = await preds.array();

    Plotly.newPlot(id, [
        {
            x: data.map(d => d.x),
            y: data.map(d => d.y),
            mode: 'markers',
            name: label + ' Real',
            marker: { color: 'gray' }
        },
        {
            x: data.map(d => d.x),
            y: predYs.map(p => p[0]),
            mode: 'lines',
            name: label + ' Prediction',
            line: { color: 'green' }
        }
    ], {
        title: `${label}: Prediction vs. Real`,
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    });

    const ys = tf.tensor2d(data.map(d => [d.y]));
    const mse = tf.losses.meanSquaredError(ys, preds).dataSync()[0];
    return mse.toFixed(4);
}

async function runAll() {
    const cleanData = generateData(100, false);
    const noisyData = generateData(100, true);
    const { train: cleanTrain, test: cleanTest } = splitData(cleanData);
    const { train: noisyTrain, test: noisyTest } = splitData(noisyData);

    plotScatter("plot-data-clean", "Unverrauschte Daten", cleanTrain, cleanTest);
    plotScatter("plot-data-noisy", "Verrauschte Daten", noisyTrain, noisyTest);

    const modelClean = await trainModel(cleanTrain, 100);
    const mseCleanTrain = await evaluateAndPlot(modelClean, "prediction-clean-train", cleanTrain, "Train Clean");
    const mseCleanTest = await evaluateAndPlot(modelClean, "prediction-clean-test", cleanTest, "Test Clean");
    document.getElementById("loss-clean").innerText = `Loss Train: ${mseCleanTrain}, Test: ${mseCleanTest}`;

    const modelBest = await trainModel(noisyTrain, 100);
    const mseBestTrain = await evaluateAndPlot(modelBest, "prediction-best-train", noisyTrain, "Train Best");
    const mseBestTest = await evaluateAndPlot(modelBest, "prediction-best-test", noisyTest, "Test Best");
    document.getElementById("loss-best").innerText = `Loss Train: ${mseBestTrain}, Test: ${mseBestTest}`;

    const modelOverfit = await trainModel(noisyTrain, 500);
    const mseOverfitTrain = await evaluateAndPlot(modelOverfit, "prediction-overfit-train", noisyTrain, "Train Overfit");
    const mseOverfitTest = await evaluateAndPlot(modelOverfit, "prediction-overfit-test", noisyTest, "Test Overfit");
    document.getElementById("loss-overfit").innerText = `Loss Train: ${mseOverfitTrain}, Test: ${mseOverfitTest}`;
}

runAll();