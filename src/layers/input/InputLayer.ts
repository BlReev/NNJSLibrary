import Assertion from "../../utils/Assertion";
import Utils from "../../utils/Utils";
import GradientHolder from "../../GradientHolder";
import Layer from "../Layer";

export default class InputLayer extends Layer {
  passInput(input: GradientHolder): Layer {
    Assertion.assert(
      Utils.shapeEquals(this.inputShape, input.shape),
      "Input Shape must match the layer's input shape!"
    );

    this.output = input;

    return this;
  }
}
