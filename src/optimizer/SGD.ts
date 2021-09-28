import GradientHolder from "GradientHolder";
import Optimizer from "./Optimizer";
export default class SGD extends Optimizer {
  optimize(trainableVariable: GradientHolder, batchSize: number): void {
    for (let index = 0; index < trainableVariable.gradv.length; index++) {
      trainableVariable.output[index] -=
        this.learningRate * (trainableVariable.gradv[index] / batchSize);
    }
  }
}
