import Tensor from "../../Tensor";
import Assertion from "../../utils/Assertion";
import Utils from "../../utils/Utils";
import BackPropagationNode from "../BackPropagationNode";
import BinaryOperationNeuron from "../BinaryOperationNeuron";
import GradientHolder from "../../GradientHolder";
import AddPropagation from "../operations/AddPropagation";
import MatMulPropagation from "../operations/MatMulPropagation";
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
