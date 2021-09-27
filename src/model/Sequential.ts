import Model from "./Model";
import GradientHolder from "../GradientHolder";
import InputLayer from "../layers/input/InputLayer";

export default class Sequential extends Model {
  forward(inputs: GradientHolder): void {
    if (this.layers[0] instanceof InputLayer) {
      this.out = this.layers[0].passInput(inputs);

      for (let index = 1; index < this.layers.length; index++) {
        this.out = this.layers[index].feedForward(this.out);
      }
    }
  }
}
