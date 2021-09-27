const fs = require('fs');
const { model } = require('./ClassifierModel.js');
const { default: Tensor } = require('./Tensor.js');

var dataFileBuffer = fs.readFileSync(__dirname + '/train-images-idx3-ubyte.gz');
var labelFileBuffer = fs.readFileSync(
  __dirname + '/train-labels-idx1-ubyte.gz'
);
const pixelValues = [];

for (let image = 0; image <= 59999; image++) {
  let pixels = [];

  for (let x = 0; x <= 27; x++) {
    for (let y = 0; y <= 27; y++) {
      pixels.push(dataFileBuffer[image * 28 * 28 + (x + y * 28) + 15]);
    }
  }

  let imageData = {};
  imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels;

  pixelValues.push(imageData);
}

for (const imageData of pixelValues) {
  for (const label in imageData) {
    const inputs = new Tensor(28, 28, 1);
    inputs.out = imageData[label];
    model.train(inputs, label);
  }
}

// TODO: Accept user input
