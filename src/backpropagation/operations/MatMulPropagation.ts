import Tensor from "../../Tensor";
import Utils from "../../utils/Utils";
import BinaryOperationNeuron from "../BinaryOperationNeuron";

export default class MatMulPropagation extends BinaryOperationNeuron {
  forwardPass(): Tensor {
    return Utils.matmul(this.firstOperand, this.secondOperand);
  }

  applyFirstGradient(): void {
    const rows = this.firstOperand.shape[0];
    const cols = this.secondOperand.shape[1];
    const x = this.secondOperand.shape[0];

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        for (let dim = 0; dim < x; dim++) {
          const multiplyGradient = this.gradv[cols * row + col];

          this.firstOperand.gradv[x * row + dim] +=
            this.secondOperand.out[cols * dim + col] * multiplyGradient;
        }
      }
    }
  }

  applySecondGradient(): void {
    const rows = this.firstOperand.shape[0];
    const cols = this.secondOperand.shape[1];
    const x = this.firstOperand.shape[1];

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        for (let dim = 0; dim < x; dim++) {
          const multiplyGradient = this.gradv[cols * row + col];

          this.secondOperand.gradv[cols * dim + col] +=
            this.firstOperand.out[x * row + dim] * multiplyGradient;
        }
      }
    }
  }
}
