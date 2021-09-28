const fs = require("fs");
const { model } = require("./ClassifierModel.js");
const { default: Tensor } = require("./Tensor.js");
const { default: ModelTrainer } = require("./trainer/ModelTrainer.js");
const { model: testModel } = require("./TestModel.js");

const TEST = true;

const inputs = [];
const labels = [];
const epochs = 1;

if (!TEST) {
  var dataFileBuffer = fs.readFileSync(__dirname + "/train-images-idx3-ubyte");
  var labelFileBuffer = fs.readFileSync(__dirname + "/train-labels-idx1-ubyte");

  for (let image = 0; image <= 59999; image++) {
    let pixels = [];

    for (let x = 0; x <= 27; x++) {
      for (let y = 0; y <= 27; y++) {
        pixels.push(dataFileBuffer[image * 28 * 28 + (x + y * 28) + 15]);
      }
    }

    labels[image] = labelFileBuffer[image + 8];
    inputs[image] = pixels;
  }

  const trainer = new ModelTrainer(model, model.learningRate, 20);

  for (let epoch = 1; epoch <= epochs; ++epoch) {
    trainer.trainEpoch(inputs, labels);
  }

  // TODO: Accept user input
} else {
  for (let inputIndex = 0; inputIndex <= 59999; inputIndex++) {
    const n = Math.floor(Math.random() * 10);
    labels[inputIndex] = n;
    inputs[inputIndex] = [n / 10];
  }
  const trainer = new ModelTrainer(testModel, testModel.learningRate, 20);
  for (let epoch = 1; epoch <= epochs; ++epoch) {
    trainer.trainEpoch(inputs, labels);
  }
}
