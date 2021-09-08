import Tensor from "./Tensor";

export default class EagerTensor extends Tensor {
  constructor(
    n: number,
    d: number,
    inputs: number[],
    grad_required: boolean = true
  ) {
    super(n, d, grad_required);

    Object.defineProperty(this, "out", {
      value: inputs,
      writable: false,
      enumerable: true,
      configurable: true,
    });
  }
}
