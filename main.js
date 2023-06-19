const tf = require('@tensorflow/tfjs-node');
const { Table } = require('console-table-printer');

////////////// Parameters //////////////
const dotsNumber = 20; // Number of input points
const xNumber = 18; // Number of input parameters
const testDotsNumber = 10; // Number of test points
const epochsNumber = 5000; // Number of training epochs

////////////// Define input and output data //////////////
// Generate random input data
const xData = tf.randomUniform([dotsNumber, xNumber]);
// Generate random weights
const wDataRaw = tf.randomUniform([xNumber]);
const wData = wDataRaw.round(2);
// Calculate the output data based on the input data and weights
const yData = xData.dot(wData.reshape([xNumber, 1]));

// Generate test data and expected output
const xNew = tf.randomUniform([testDotsNumber, xNumber]);
const expectedNewY = xNew.dot(wData.reshape([xNumber, 1]));

////////////// Define the model parameters //////////////
const inputDim = xNumber;
const hiddenDim = xNumber;
const outputDim = 1;

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.dense({ units: hiddenDim, inputShape: [inputDim], activation: 'relu' }));
model.add(tf.layers.dense({ units: outputDim }));

// Compile the model
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

////////////// Train the model //////////////
async function trainModel() {
  const table = new Table();
  table.addColumns(['Epoch', 'Loss']);

  for (let i = 0; i < epochsNumber; i++) {
    const history = await model.fit(xData, yData, { epochs: 1 });
    const loss = history.history.loss[0];

    if (i % 100 === 0) {
      console.log(`Step ${i}, Loss: ${loss}`);
    }

    table.addRow({ Epoch: i + 1, Loss: loss.toFixed(4) });
  }

  // Use the trained model to make predictions on new data
  const yPred = model.predict(xNew);

  // Print the training results
  console.log('\nTraining Results:');
  table.printTable();

  // Print the prediction results for comparison
  const resultTable = new Table();
  resultTable.addRows(
    yPred.arraySync().map((pred, i) => {
      const expected = expectedNewY.arraySync()[i][0];
      const delta = Math.abs((pred[0] - expected) / expected) * 100;
      return { 'Predicted Y': pred[0], 'Expected Y': expected, 'Delta %': delta.toFixed(2) };
    })
  );
  console.log('\nPrediction Results:');
  resultTable.printTable();
}

trainModel();
