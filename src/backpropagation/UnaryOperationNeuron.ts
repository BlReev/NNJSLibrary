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

  abstract applyFirstGradient(target?: number): void;

  propagateBackwards(target?: number): void {
    if (this.firstOperand.grad_required) {
      this.applyFirstGradient(target);

      if (this.firstOperand instanceof PropagationOperation) {
        this.firstOperand.propagateBackwards(target);
      }
    }
  }
}
