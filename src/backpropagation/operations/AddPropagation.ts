import Tensor from "../../Tensor";
import BinaryOperationNeuron from "../BinaryOperationNeuron";

export default class AddPropagation extends BinaryOperationNeuron {
  forwardPass(): Tensor {
    const length = this.firstOperand.output.length;
    const tensor: Tensor = new Tensor(1, 1, length, true);

    for (let i = 0; i < length; i++) {
      tensor.output[i] =
        this.firstOperand.output[i] + this.secondOperand.output[i];
    }

    return tensor;
  }

  applyFirstGradient(): void {
    this.firstOperand.grad(this.gradv);
  }
  applySecondGradient(): void {
    this.secondOperand.grad(this.gradv);
  }
}
