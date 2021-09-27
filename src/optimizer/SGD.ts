import OptimizableLayer from "../layers/OptimizableLayer";
import Utils from "../utils/Utils";
import Optimizer from "./Optimizer";
export default class SGD extends Optimizer {
  optimize(): void {
    for (const layer of this.model.layers) {
      if (layer instanceof OptimizableLayer) {
        layer.optimize(this.learningRate);
      }
    }
  }
  reset(): void {
    for (const layer of this.model.layers) {
      if (Object.prototype.hasOwnProperty.call(layer, "W")) {
        let len_w = layer.W.output.length;
        let len_b = layer.b.output.length;

        layer.W.grad(Utils.buildOneDimensionalArray(len_w, () => 0));
        layer.b.grad(Utils.buildOneDimensionalArray(len_b, () => 0));
      }
    }
  }
}
