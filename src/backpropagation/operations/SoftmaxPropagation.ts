import Tensor from "../../Tensor";
import Utils from "../../utils/Utils";
import UnaryOperationNeuron from "../UnaryOperationNeuron";

export default class SoftmaxPropagation extends UnaryOperationNeuron {
  forwardPass(): Tensor {
    const input = this.firstOperand;
    const length = input.shape[1];
    const tensor: Tensor = new Tensor(1, length, true);
    const activation = input.out;
    const activationMax = Math.max(...input.out);

    const expArray = Utils.buildOneDimensionalArray(length, () => 0);

    let expSum = 0.0;
    for (let expIndex = 0; expIndex < length; expIndex++) {
      const e = Math.exp(activation[expIndex] - activationMax);
      expSum += e;
      expArray[expIndex] = e;
    }

    for (const index = 0; index < length; index++) {
      expArray[index] /= expSum;
      tensor.out[index] = expArray[index];
    }

    return tensor;
  }

  applyFirstGradient(): void {
    const input = this.firstOperand;

    for (var index = 0; index < this.items.shape[1]; index++) {
      var indicator = index === y ? 1.0 : 0.0;
      var mul = -(indicator - this.out[index]);
      input.gradv[index] = mul;
    }
  }
}