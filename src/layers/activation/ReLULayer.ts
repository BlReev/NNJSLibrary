import GradientHolder from "../../GradientHolder";
import Layer from "../Layer";
import ActivationLayer from "./ActivationLayer";
import Utils from "../../utils/Utils";
import Tensor from "../../Tensor";

export default class ReLULayer extends ActivationLayer {
  feedForward(inputs: Layer): Layer {
    super.feedForward(inputs);

    this.input = inputs;

    this.output = new Tensor(
      inputs.outputShape[0],
      inputs.outputShape[1],
      inputs.outputShape[2]
    );

    for (const index in inputs.output.output) {
      if (inputs.output.output[index] >= 0)
        this.output.output[index] = inputs.output.output[index];
    }

    return this;
  }

  propagateBackwards(): void {
    const inputs: Layer = this.input;
    const outputLength = inputs.W.output.length;
    inputs.W.gradv = Utils.buildOneDimensionalArray(outputLength, () => 0);

    for (var i = 0; i < outputLength; i++) {
      if (inputs.output[i] <= 0) this.W.gradv[i] = 0;
      else this.W.gradv[i] = inputs.W.gradv[i];
    }
  }
}
