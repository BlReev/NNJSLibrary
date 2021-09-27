import GradientHolder from "./GradientHolder";
import Utils from "./utils/Utils";

export default class Tensor extends GradientHolder {
  constructor(w: number, h: number, d: number = 1) {
    super();

    this.shape = [w, h, d];
    this.output = Utils.buildOneDimensionalArray(w * h * d, () => 0);
    this.grad(Utils.buildOneDimensionalArray(w * h * d, () => 0));
  }

  fillGaussianRandom(mean: number, standardDeviation: number) {
    Utils.randnArray(this.output, mean, standardDeviation);

    return this;
  }
}
