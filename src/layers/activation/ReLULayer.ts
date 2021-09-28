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

    for (var index = 0; index < outputLength; index++) {
      if (this.output[index] <= 0) inputs.W.gradv[index] = 0;
      else inputs.W.gradv[index] = this.W.gradv[index];
    }
  }
}
