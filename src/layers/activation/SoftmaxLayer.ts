import GradientHolder from "../../GradientHolder";
import Layer from "../Layer";
import ActivationLayer from "./ActivationLayer";
import Tensor from "../../Tensor";
import Utils from "../../utils/Utils";

export default class SoftmaxLayer extends ActivationLayer {
  exponents: number[];

  feedForward(inputs: Layer): Layer {
    super.feedForward(inputs);

    this.input = inputs;

    const length = this.outputShape[2];
    const activation = inputs.output.output;
    const activationMax = Math.max(...activation);

    const expArray = Utils.buildOneDimensionalArray(length, () => 0);

    let expSum = 0.0;
    for (let expIndex = 0; expIndex < length; ++expIndex) {
      const exp = Math.exp(activation[expIndex] - activationMax);
      expSum += exp;
      expArray[expIndex] = exp;
    }

    for (let index = 0; index < length; index++) {
      expArray[index] /= expSum;
      this.output.output[index] = expArray[index];
    }

    this.exponents = expArray;
    return this;
  }

  propagateBackwards(correctLabel: number = 0): number {
    const inputs: Layer = this.input;
    inputs.W.gradv = Utils.buildOneDimensionalArray(
      inputs.W.output.length,
      () => 0
    );

    for (var index = 0; index < this.outputShape[2]; index++) {
      let indicator = index === correctLabel ? 1.0 : 0.0;
      let mul = this.exponents[index] - indicator;
      inputs.W.gradv[index] = mul;
    }

    return -Math.log(this.exponents[correctLabel]);
  }
}
