import GradientHolder from "./GradientHolder";
import Utils from "./utils/Utils";

export default class Tensor extends GradientHolder {
  constructor(
    w: number,
    h: number,
    d: number = 1,
    grad_required: boolean = true
  ) {
    super();

    this.shape = [w, h, d];
    this.output = Utils.buildOneDimensionalArray(w * h * d, () => 0);
    this.grad(Utils.buildOneDimensionalArray(w * h * d, () => 0));
    this.grad_required = grad_required;
  }

  fillGaussianRandom(mean: number, standardDeviation: number) {
    Utils.randnArray(this.output, mean, standardDeviation);

    return this;
  }
}
