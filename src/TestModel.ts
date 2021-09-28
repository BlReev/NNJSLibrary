import Sequential from "./model/Sequential";
import LossType from "./loss/LossType";
import InputLayer from "./layers/input/InputLayer";
import FullyConnectedLayer from "./layers/dotproduct/fullyconnected/FullyConnectedLayer";
import SoftmaxLayer from "./layers/activation/SoftmaxLayer";
import Adadelta from "./optimizer/Adadelta";

const learningRate = 1.0;

// Create and compile the model
export const model = new Sequential(
  [
    new InputLayer(1, 1, 1),
    new FullyConnectedLayer(1, 1, 1, 10),
    new SoftmaxLayer(1, 1, 10),
  ],
  {
    optimizer: new Adadelta(learningRate),
    loss: LossType.CrossEntropy,
    learningRate,
  }
);
