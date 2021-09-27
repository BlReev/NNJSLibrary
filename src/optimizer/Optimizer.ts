import GradientHolder from "GradientHolder";
import OptimizableLayer from "layers/OptimizableLayer";
import Utils from "utils/Utils";
import Model from "../model/Model";
export default abstract class Optimizer {
  model: Model;
  learningRate: number;

  constructor(model: Model, learningRate: number) {
    this.model = model;
    this.learningRate = learningRate;
  }

  abstract optimize(trainableVariable: GradientHolder): void;

  reset(trainableVariable: GradientHolder): void {
    trainableVariable.grad(
      Utils.buildOneDimensionalArray(trainableVariable.gradv.length, () => 0)
    );
  }
}
