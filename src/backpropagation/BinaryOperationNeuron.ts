import Tensor from "../Tensor";
import GradientHolder from "../GradientHolder";
import PropagationOperation from "./PropagationOperation";

export default abstract class BinaryOperationNeuron extends PropagationOperation {
  firstOperand: GradientHolder;
  secondOperand: GradientHolder;

  constructor(firstOperand: GradientHolder, secondOperand: GradientHolder) {
    super(firstOperand, secondOperand);
  }

  initializeOperands(...operands: GradientHolder[]): void {
    this.firstOperand = operands[0];
    this.secondOperand = operands[1];
  }

  abstract applyFirstGradient(): void;

  abstract applySecondGradient(): void;

  propagateBackwards(): void {
    if (this.firstOperand.grad_required) {
      this.applyFirstGradient();

      if (this.firstOperand instanceof PropagationOperation) {
        this.firstOperand.propagateBackwards();
      }
    }

    if (this.secondOperand.grad_required) {
      this.applySecondGradient();

      if (this.secondOperand instanceof PropagationOperation)
        this.secondOperand.propagateBackwards();
    }
  }
}
