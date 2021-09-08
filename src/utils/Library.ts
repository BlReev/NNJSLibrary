import EagerTensor from "../EagerTensor";
import Tensor from "../Tensor";
import Assertion from "./Assertion";

export default {
  // Returns a tensor which output can be changed
  variable(n: number, d: number, inputs: number[]): Tensor {
    Assertion.assert(
      n * d === inputs.length,
      "Input length does not match the shape"
    );

    const tensor = new Tensor(n, d);
    tensor.setOut(inputs);

    return tensor;
  },
  // Returns a constant tensor which output can't be changed
  constant(n: number, d: number, inputs: number[]): EagerTensor {
    Assertion.assert(
      n * d === inputs.length,
      "Input length does not match the shape"
    );

    return new EagerTensor(n, d, inputs);
  },
};
