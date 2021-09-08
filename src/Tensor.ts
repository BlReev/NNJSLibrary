import GradientHolder from "./GradientHolder";
import Utils from "./utils/Utils";

export default class Tensor extends GradientHolder {
  constructor(n: number, d: number, grad_required: boolean = true) {
    super();

    this.shape = [n, d];
    this.out = Utils.buildOneDimensionalArray(n * d, () => 0);
    this.grad(Utils.buildOneDimensionalArray(n * d, () => 0));
    this.grad_required = grad_required;
  }

  fillGaussianRandom(mean: number, standardDeviation: number) {
    Utils.randnArray(this.out, mean, standardDeviation);

    return this;
  }
}
