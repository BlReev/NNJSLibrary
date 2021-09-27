import GradientHolder from "GradientHolder";
import OptimizableLayer from "../layers/OptimizableLayer";
import Utils from "../utils/Utils";
import Optimizer from "./Optimizer";
export default class SGD extends Optimizer {
  /*
   * TODO
   *  Implement different optimizers
   *  The only optimizer supported is
   *  Stochastic gradient descent.
   */
  optimize(trainableVariable: GradientHolder): void {
    for (let index = 0; index < trainableVariable.gradv.length; index++) {
      trainableVariable.output[index] -=
        this.learningRate * trainableVariable.gradv[index];
    }
  }
}
