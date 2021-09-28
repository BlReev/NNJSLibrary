import GradientHolder from "GradientHolder";
import Utils from "../utils/Utils";
import Model from "../model/Model";
export default abstract class Optimizer {
  learningRate: number;

  constructor(learningRate: number) {
    this.learningRate = learningRate;
  }

  abstract optimize(
    trainableVariable: GradientHolder,
    batchSize: number,
    index?: number
  ): void;

  reset(trainableVariable: GradientHolder): void {
    trainableVariable.grad(
      Utils.buildOneDimensionalArray(trainableVariable.gradv.length, () => 0)
    );
  }
}
