import Tensor from "../../Tensor";
import BinaryOperationNeuron from "../BinaryOperationNeuron";

export default class AddPropagation extends BinaryOperationNeuron {
  forwardPass(): Tensor {
    const length = this.firstOperand.out.length;
    const tensor: Tensor = new Tensor(1, length, true);

    for (let i = 0; i < length; i++) {
      tensor.out[i] = this.firstOperand.out[i] + this.secondOperand.out[i];
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
