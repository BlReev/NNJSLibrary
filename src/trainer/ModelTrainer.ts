import GradientHolder from "GradientHolder";
import Model from "../model/Model";

interface TrainResult {
  forwardTime: number;
  backwardTime: number;
  costLoss: number;
  loss: number;
}

export default class ModelTrainer {
  private model: Model;
  private learningRate: number;
  private trainBatchSize: number;
  private currentBatchSize: number;

  constructor(model: Model, learningRate: number, trainBatchSize: number) {
    this.model = model;
    this.learningRate = learningRate;
    this.trainBatchSize = trainBatchSize;
    this.currentBatchSize = 0;
  }

  train(inputs: GradientHolder, targetLabel: number): TrainResult {
    const startForward = Date.now();
    this.model.guess(inputs);
    const forwardTime = Date.now() - startForward;
    const startBackward = Date.now();
    const costLoss: number = this.model.backward(targetLabel);
    const backwardTime = Date.now() - startBackward;

    if (++this.currentBatchSize % this.trainBatchSize === 0) {
      /* this.optimize(); */
    }

    return {
      forwardTime,
      backwardTime,
      costLoss,
      loss: costLoss /* TODO: Implement Weight Decay Loss */,
    };
  }
}
