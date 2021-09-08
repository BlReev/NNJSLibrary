import GradientHolder from "../../GradientHolder";
import Layer from "../Layer";
import ActivationLayer from "./ActivationLayer";
import SoftmaxPropagation from "../../backpropagation/operations/SoftmaxPropagation";
import PropagationOperation from "../../backpropagation/PropagationOperation";

export default class SoftmaxLayer extends ActivationLayer {
  feedForward(inputs: GradientHolder): Layer {
    this.items = new SoftmaxPropagation(inputs);
    this.shape = inputs.shape;
    this.out = this.items.out;
    this.gradv = this.items.gradv;

    return this;
  }

  propagateBackwards(correctLabel: number = 0): void {
    this.items.grad(this.gradv);

    if (this.items instanceof PropagationOperation) {
      if (this.items instanceof SoftmaxPropagation) {
        this.items.propagateBackwards(correctLabel);
      } else {
        this.items.propagateBackwards();
      }
    }
  }
}
