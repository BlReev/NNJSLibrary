import Utils from "../../../utils/Utils";
import OptimizableLayer from "../../OptimizableLayer";
import Tensor from "../../../Tensor";
import Layer from "../../Layer";

export default class FullyConnectedLayer extends OptimizableLayer {
  filters: Tensor[];

  constructor(w, h, d, neuronCount, bias: number = 0) {
    super(1, 1, neuronCount);

    this.inputShape = [w, h, d];
    this.filters = [];

    for (var filterIndex = 0; filterIndex < d; filterIndex++) {
      this.filters.push(new Tensor(1, 1, w * h * d));
    }
  }

  feedForward(inputs: Layer): Layer {
    super.feedForward(inputs);

    const weights = inputs.output;
    const inputCount =
      this.inputShape[0] * this.inputShape[1] * this.inputShape[2];

    for (let i = 0; i < this.outputShape[2]; i++) {
      let weight = 0.0;
      const filterWeights = this.filters[i].output;
      for (let weightIndex = 0; weightIndex < inputCount; weightIndex++) {
        weight += weights[weightIndex] * filterWeights[weightIndex];
      }
      weight += this.b.output[i];
      this.output[i] = weight;
    }

    return this;
  }

  propagateBackwards() {
    const input: Layer = this.input;
    input.W.gradv = Utils.buildOneDimensionalArray(input.W.output.length);

    for (
      let currentDepth: number = 0;
      currentDepth < this.outputShape[2];
      currentDepth++
    ) {
      const filter: Tensor = this.filters[currentDepth];
      const chainedGradient = this.input.W.gradv[currentDepth];

      for (var index = 0; index < this.inputShape[0]; index++) {
        input.W.gradv[index] += filter.output[index] * chainedGradient;
        filter.gradv[index] += input.output[index] * chainedGradient;
      }

      this.b.gradv[currentDepth] += chainedGradient;
    }
  }

  optimize(learningRate: number): void {
    super.optimize(learningRate);

    for (const filter of this.filters) {
      for (let index = 0; index < filter.gradv.length; index++) {
        filter.output[index] -= learningRate * filter.gradv[index];
      }
    }
  }
}
