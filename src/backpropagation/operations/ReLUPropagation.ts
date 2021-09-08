import Tensor from "../../Tensor";
import UnaryOperationNeuron from "../UnaryOperationNeuron";

export default class ReLUPropagation extends UnaryOperationNeuron {
  forwardPass(): Tensor {
    console.log(this);
    const length = this.firstOperand.out.length;
    const tensor: Tensor = new Tensor(
      this.firstOperand.shape[0],
      this.firstOperand.shape[1],
      true
    );

    for (let index = 0; index < length; index++) {
      tensor.out[index] = Math.max(0, this.firstOperand.out[index]);
    }

    return tensor;
  }

  applyFirstGradient(): void {
    for (let index = 0; index < this.firstOperand.out.length; index++) {
      this.firstOperand.gradv[index] =
        this.firstOperand.out[index] > 0 ? this.gradv[index] : 0.0;
    }
  }
}
