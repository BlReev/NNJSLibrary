import Layer from "./Layer";

export default abstract class OptimizableLayer extends Layer {
  /*
   * TODO
   *  Implement different optimizers
   *  The only optimizer supported is
   *  Stochastic gradient descent.
   */
  optimize(learningRate: number): void {
    for (let index = 0; index < this.W.output.length; index++) {
      this.W.output[index] -= learningRate * this.W.gradv[index];
    }

    for (let index = 0; index < this.b.output.length; index++) {
      this.b.output[index] -= learningRate * this.b.gradv[index];
    }
  }
}
