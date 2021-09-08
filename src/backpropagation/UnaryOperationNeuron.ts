import Tensor from "../Tensor";
import GradientHolder from "../GradientHolder";
import PropagationOperation from "./PropagationOperation";

export default abstract class UnaryOperationNeuron extends PropagationOperation {
  firstOperand: GradientHolder;

  constructor(firstOperand: GradientHolder) {
    super(firstOperand);
  }

  initializeOperands(...operands: GradientHolder[]): void {
    this.firstOperand = operands[0];
  }

  abstract applyFirstGradient(): void;

  propagateBackwards(): void {
    if (this.firstOperand.grad_required) {
      this.applyFirstGradient();

      if (this.firstOperand instanceof PropagationOperation) {
        this.firstOperand.propagateBackwards();
      }
    }
  }
}
