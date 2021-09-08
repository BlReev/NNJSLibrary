import ActivationType from "./ActivationType";
import GradientHolder from "../../../GradientHolder";
import Layer from "../Layer";
import ReLUPropagation from "../../operations/ReLUPropagation";
import PropagationOperation from "../../PropagationOperation";

export default abstract class ActivationLayer extends Layer {
  constructor(input: GradientHolder) {
    super(input.shape[0], input.shape[1]);
  }

  abstract feedForward(inputs: GradientHolder): Layer;
}
