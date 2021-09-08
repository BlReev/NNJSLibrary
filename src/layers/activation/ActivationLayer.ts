import GradientHolder from "../../GradientHolder";
import Layer from "../Layer";

export default abstract class ActivationLayer extends Layer {
  constructor(input: GradientHolder) {
    super(input.shape[0], input.shape[1]);
  }

  abstract feedForward(inputs: GradientHolder): Layer;
}
