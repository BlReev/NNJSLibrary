import ActivationLayer from "../layers/activation/ActivationLayer";
import Layer from "../layers/Layer";
import Model from "../model/Model";
export default class Loss {
  layer: Layer;
  out: number;
  target: number;

  constructor(target: number, model: Model) {
    this.layer = model.out;
    this.out = -Math.log(model.out.output[target]);
    this.target = target;
  }

  startPropagation(): void {
    if (this.layer instanceof ActivationLayer) {
      this.layer.propagateBackwards(this.target);
    } else {
      this.layer.propagateBackwards();
    }
  }
}
