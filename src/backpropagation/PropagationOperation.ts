import Tensor from "../Tensor";
import BackPropagationNode from "./BackPropagationNode";
import GradientHolder from "../GradientHolder";

export default abstract class PropagationOperation
  extends GradientHolder
  implements BackPropagationNode
{
  input: Tensor;

  constructor(...operands: GradientHolder[]) {
    super();
    this.grad_required = true;
    this.initializeOperands(...operands);
    this.input = this.forwardPass();
    this.output = this.input.output;
    this.grad(this.input.gradv);
  }

  abstract initializeOperands(...operands: GradientHolder[]): void;

  abstract forwardPass(): Tensor;

  abstract propagateBackwards(target?: number): void;
}
