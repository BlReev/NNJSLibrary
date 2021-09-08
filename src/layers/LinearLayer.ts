import Assertion from "../utils/Assertion";
import GradientHolder from "../GradientHolder";
import AddPropagation from "../backpropagation/operations/AddPropagation";
import MatMulPropagation from "../backpropagation/operations/MatMulPropagation";
import Layer from "./Layer";

export default class LinearLayout extends Layer {
  feedForward(inputs: GradientHolder): Layer {
    Assertion.assert(
      inputs.out.length === this.shape[0],
      "Feed Forward was given an invalid input count."
    );

    const matmulOperation = new MatMulPropagation(inputs, this.W);
    this.items = new AddPropagation(matmulOperation, this.b);
    this.shape = matmulOperation.shape;
    this.out = this.items.out;
    this.grad(this.items.gradv);

    return this;
  }
}
