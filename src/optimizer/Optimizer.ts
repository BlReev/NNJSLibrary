import Model from '../model/Model';
export default abstract class Optimizer {
  model: Model;
  learningRate: number;

  constructor(model: Model, learningRate: number) {
    this.model = model;
    this.learningRate = learningRate;
  }

  abstract optimize(): void;

  abstract reset(): void;
}
