import Utils from "../../../utils/Utils";
import OptimizableLayer from "../../OptimizableLayer";
import Tensor from "../../../Tensor";
import Layer from "../../Layer";
import GradientHolder from "GradientHolder";

export default class FullyConnectedLayer extends OptimizableLayer {
  filters: Tensor[];

  constructor(
    w: number,
    h: number,
    d: number,
    neuronCount: number,
    bias: number = 0
  ) {
    super(1, 1, neuronCount);

    this.inputShape = [w, h, d];
    this.filters = [];

    for (var filterIndex = 0; filterIndex < neuronCount; filterIndex++) {
      this.filters.push(new Tensor(1, 1, w * h * d));
    }

    this.b.output = Utils.buildOneDimensionalArray(
      this.b.output.length,
      () => bias
    );
  }

  feedForward(inputs: Layer): Layer {
    super.feedForward(inputs);

    const weights = inputs.output.output;
    const inputCount =
      this.inputShape[0] * this.inputShape[1] * this.inputShape[2];

    for (let depth = 0; depth < this.outputShape[2]; depth++) {
      let weight = 0.0;
      const filterWeights = this.filters[depth].output;
      for (let weightIndex = 0; weightIndex < inputCount; weightIndex++) {
        weight += weights[weightIndex] * filterWeights[weightIndex];
      }
      weight += this.b.output[depth];
      this.output.output[depth] = weight;
    }

    return this;
  }

  propagateBackwards() {
    const input: Layer = this.input;
    const inputCount =
      this.inputShape[0] * this.inputShape[1] * this.inputShape[2];
    input.W.gradv = Utils.buildOneDimensionalArray(input.W.output.length);

    for (
      let currentDepth: number = 0;
      currentDepth < this.outputShape[2];
      currentDepth++
    ) {
      const filter: Tensor = this.filters[currentDepth];
      const chainedGradient = this.W.gradv[currentDepth];
      for (let index = 0; index < inputCount; index++) {
        input.W.gradv[index] += filter.output[index] * chainedGradient;
        filter.gradv[index] += input.output.output[index] * chainedGradient;
      }

      this.b.gradv[currentDepth] += chainedGradient;
    }
  }

  getTrainableVariables(): GradientHolder[] {
    return [...this.filters, this.b];
  }
}
