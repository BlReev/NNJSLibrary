import GradientHolder from "../../../GradientHolder";
import ReLUPropagation from "../../operations/ReLUPropagation";
import Layer from "../Layer";
import ActivationLayer from "./ActivationLayer";

export default class ReLULayer extends ActivationLayer {
  feedForward(inputs: GradientHolder): Layer {
    this.items = new ReLUPropagation(inputs);
    this.shape = inputs.shape;
    this.out = this.items.out;
    this.gradv = this.items.gradv;

    return this;
  }
}
