import Sequential from "./model/Sequential";
import LossType from "./loss/LossType";
import InputLayer from "./layers/input/InputLayer";
import ConvolutionLayer from "./layers/dotproduct/conv/ConvolutionLayer";
import ReLULayer from "./layers/activation/ReLULayer";
import MaxPoolingLayer from "./layers/pooling/MaxPoolingLayer";
import FullyConnectedLayer from "./layers/dotproduct/fullyconnected/FullyConnectedLayer";
import SoftmaxLayer from "./layers/activation/SoftmaxLayer";
import Adadelta from "./optimizer/Adadelta";

const learningRate = 1.0;

// Create and compile the model
export const model = new Sequential(
  [
    new InputLayer(28, 28, 1),
    new ConvolutionLayer(28, 28, 1, 2, [5, 5, 1], 8, 1),
    new ReLULayer(28, 28, 8),
    new MaxPoolingLayer(28, 28, 8, [6, 6, 1], 0, 2),
    new ConvolutionLayer(12, 12, 8, 2, [5, 5, 8], 16, 1),
    new ReLULayer(12, 12, 16),
    new MaxPoolingLayer(12, 12, 16, [3, 3, 1], 0, 3),
    new FullyConnectedLayer(4, 4, 16, 10),
    new SoftmaxLayer(1, 1, 10),
  ],
  {
    optimizer: new Adadelta(learningRate),
    loss: LossType.CrossEntropy,
    learningRate,
  }
);
