import GradientHolder from "GradientHolder";
import Layer from "./Layer";

export default abstract class OptimizableLayer extends Layer {
  getTrainableVariables(): GradientHolder[] {
    return [this.W, this.b];
  }
}
