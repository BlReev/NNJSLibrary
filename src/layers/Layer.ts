import Tensor from "../Tensor";
import GradientHolder from "../GradientHolder";
import Utils from "../utils/Utils";
import Assertion from "../utils/Assertion";

export default abstract class Layer {
  input: Layer;
  inputShape: number[];
  output: GradientHolder;
  outputShape: number[];
  W: Tensor;
  b: Tensor;

  constructor(w: number, h: number, d: number) {
    this.outputShape = [w, h, d];
    this.inputShape = [w, h, d];
    this.output = new Tensor(w, h, d);
    this.W = new Tensor(w, h, d).fillGaussianRandom(0, 0.88);
    this.b = new Tensor(1, 1, d).fillGaussianRandom(0, 0.88);
  }

  propagateBackwards(target?: number): void {
    if (this.input instanceof Layer) {
      this.input.output.grad(this.output.gradv);
      this.input.propagateBackwards(target);
    }
  }

  feedForward(input: Layer): Layer {
    Assertion.assert(
      Utils.shapeEquals(this.inputShape, input.outputShape),
      `Input Shape must match the layer's input shape!\n  Shapes:\n  Input: ${input.outputShape}\n  Layer: ${this.inputShape}`
    );

    this.input = input;

    return this;
  }

  get(row: number, col: number, depth: number) {
    return this.output.get(row, col, depth);
  }

  set(row: number, col: number, depth: number, value: number) {
    this.output.set(row, col, depth, value);
  }
}
