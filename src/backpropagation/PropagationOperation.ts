import Tensor from "../Tensor";
import BackPropagationNode from "./BackPropagationNode";
import GradientHolder from "../GradientHolder";

export default abstract class PropagationOperation
  extends GradientHolder
  implements BackPropagationNode
{
  items: Tensor;

  constructor(...operands: GradientHolder[]) {
    super();
    this.grad_required = true;
    this.initializeOperands(...operands);
    this.items = this.forwardPass();
    this.out = this.items.out;
    this.grad(this.items.gradv);
  }

  abstract initializeOperands(...operands: GradientHolder[]): void;

  abstract forwardPass(): Tensor;

  abstract propagateBackwards(): void;
}
