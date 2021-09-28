import Layer from "../layers/Layer";
import ModelOptions from "./ModelOptions";
import Optimizer from "../optimizer/Optimizer";
import GradientHolder from "../GradientHolder";
import LossType from "../loss/LossType";
import OptimizerType from "../optimizer/OptimizerType";
import SGD from "../optimizer/SGD";
import SoftmaxLayer from "../layers/activation/SoftmaxLayer";
import InputLayer from "../layers/input/InputLayer";
import OptimizableLayer from "../layers/OptimizableLayer";

interface OptimizeResult {}

export default abstract class Model {
  layers: Layer[];
  optimizer: Optimizer;
  loss: string;
  inputShape: number[];
  outputShape: number[];
  out: Layer;

  constructor(layers: Layer[], modelOptions: ModelOptions = null) {
    this.layers = layers;

    if (modelOptions !== null) {
      this.compile(modelOptions);
    }
  }

  compile(options: ModelOptions) {
    if (options.optimizer instanceof Optimizer) {
      this.optimizer = options.optimizer;
    } else {
      switch (options.optimizer) {
        case OptimizerType.SGD:
          this.optimizer = new SGD(options.learningRate);
          break;
        default:
          this.optimizer = null;
          break;
      }
    }
    this.loss = options.loss;

    const layer = this.layers[0];
    if (!(layer instanceof InputLayer)) {
      this.layers.unshift(
        new InputLayer(
          layer.inputShape[0],
          layer.inputShape[1],
          layer.inputShape[2]
        )
      );
    }

    this.inputShape = this.layers[0].inputShape;
    this.outputShape = this.layers[this.layers.length - 1].outputShape;
  }

  abstract forward(input: GradientHolder): void;

  backward(target: number): number {
    let loss: number = 0;
    const layerCount = this.layers.length;
    const lastLayer = this.layers[layerCount - 1];

    if (lastLayer instanceof SoftmaxLayer) {
      switch (this.loss) {
        case LossType.CrossEntropy:
          loss = lastLayer.propagateBackwards(target);
          for (let layerIndex = layerCount - 2; layerIndex >= 0; layerIndex--) {
            this.layers[layerIndex].propagateBackwards();
          }
          break;
        default:
          loss = null;
          break;
      }
    }

    return loss;
  }

  optimize(batchSize: number = 1): OptimizeResult {
    const trainableVariables = this.getTrainableVariables();

    for (let index = 0; index < trainableVariables.length; ++index) {
      this.optimizer.optimize(trainableVariables[index], batchSize, index);
      this.optimizer.reset(trainableVariables[index]);
    }

    return {};
  }

  getTrainableVariables(): GradientHolder[] {
    const trainableVariables: GradientHolder[] = [];

    for (const layer of this.layers) {
      if (layer instanceof OptimizableLayer) {
        trainableVariables.push(...layer.getTrainableVariables());
      }
    }

    return trainableVariables;
  }

  train(inputs: GradientHolder, target: number): number[] {
    const output = this.guess(inputs);
    this.backward(target);
    this.optimize();

    return output.output;
  }

  guess(inputs: GradientHolder) {
    this.forward(inputs);

    return this.out.output;
  }

  getPrediction() {
    var lossLayer = this.layers[this.layers.length - 1];
    var predictionArray = lossLayer.output.output;
    var maxValue = predictionArray[0];
    var maxIndex = 0;
    for (
      var predictionIndex = 1;
      predictionIndex < predictionArray.length;
      predictionIndex++
    ) {
      if (predictionArray[predictionIndex] > maxValue) {
        maxValue = predictionArray[predictionIndex];
        maxIndex = predictionIndex;
      }
    }
    return maxIndex;
  }
}
