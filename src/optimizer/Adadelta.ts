import GradientHolder from "GradientHolder";
import Utils from "../utils/Utils";
import Optimizer from "./Optimizer";
export default class Adadelta extends Optimizer {
  private rho: number;
  private epsilon: number;
  private gradSum: number[][];
  private learnSum: number[][];

  constructor(
    learningRate: number,
    rho: number = 0.95,
    epsilon: number = 0.1e-7
  ) {
    super(learningRate);

    this.rho = rho;
    this.epsilon = epsilon;
    this.gradSum = [];
    this.learnSum = [];
  }

  optimize(
    trainableVariable: GradientHolder,
    batchSize: number,
    index: number
  ): void {
    if (!this.gradSum[index] || !this.learnSum[index]) {
      this.gradSum[index] = Utils.buildOneDimensionalArray(
        trainableVariable.output.length,
        () => 0
      );
      this.learnSum[index] = Utils.buildOneDimensionalArray(
        trainableVariable.output.length,
        () => 0
      );
    }

    const gradientSum = this.gradSum[index];
    const learnSum = this.learnSum[index];
    for (let index = 0; index < trainableVariable.output.length; index++) {
      const grad = trainableVariable.gradv[index] / batchSize;
      gradientSum[index] =
        this.rho * gradientSum[index] + (1 - this.rho) * grad * grad;
      const learningGradient =
        -Math.sqrt(
          (learnSum[index] + this.epsilon) / (gradientSum[index] + this.epsilon)
        ) * grad;
      learnSum[index] =
        this.rho * learnSum[index] +
        (1 - this.rho) * learningGradient * learningGradient;
      trainableVariable.output[index] += learningGradient;
    }
  }
}
