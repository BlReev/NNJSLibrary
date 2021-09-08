import Tensor from "../../Tensor";
import GradientHolder from "../../GradientHolder";
import BackPropagationNode from "../BackPropagationNode";
import PropagationOperation from "../PropagationOperation";

export default abstract class Layer
  extends GradientHolder
  implements BackPropagationNode
{
  items: GradientHolder;
  W: Tensor;
  b: Tensor;
  feedCache: number;

  constructor(inputDimension: number, outputDimension: number) {
    super();

    this.shape = [inputDimension, outputDimension];
    this.W = new Tensor(inputDimension, outputDimension).fillGaussianRandom(
      0,
      0.88
    );
    this.b = new Tensor(outputDimension, 1).fillGaussianRandom(0, 0.88);
  }

  forwardPass(): GradientHolder {
    return this.items;
  }

  propagateBackwards(): void {
    this.items.grad(this.gradv);

    if (this.items instanceof PropagationOperation) {
      this.items.propagateBackwards();
    }
  }

  abstract feedForward(inputs: GradientHolder): Layer;
}
