import Tensor from "../../Tensor";
import Utils from "../../utils/Utils";
import PropagationOperation from "../PropagationOperation";
import UnaryOperationNeuron from "../UnaryOperationNeuron";

export default class SoftmaxPropagation extends UnaryOperationNeuron {
  forwardPass(): Tensor {
    const input = this.firstOperand;
    const length = this.shape[2];
    const tensor: Tensor = new Tensor(1, 1, length, true);
    const activation = input.output;
    const activationMax = Math.max(...input.output);

    const expArray = Utils.buildOneDimensionalArray(length, () => 0);

    let expSum = 0.0;
    for (let expIndex = 0; expIndex < length; ++expIndex) {
      const exp = Math.exp(activation[expIndex] - activationMax);
      expSum += exp;
      expArray[expIndex] = exp;
    }

    for (let index = 0; index < length; index++) {
      expArray[index] /= expSum;
      tensor.output[index] = expArray[index];
    }

    return tensor;
  }

  applyFirstGradient(target: number = 0): void {
    const input = this.firstOperand;

    for (let index = 0; index < this.shape[2]; index++) {
      const indicator = index === target ? 1.0 : 0.0;
      input.gradv[index] = this.output[index] - indicator;
    }
  }

  propagateBackwards(target: number = 0): void {
    if (this.firstOperand.grad_required) {
      this.applyFirstGradient(target);

      if (this.firstOperand instanceof PropagationOperation) {
        this.firstOperand.propagateBackwards();
      }
    }
  }
}
