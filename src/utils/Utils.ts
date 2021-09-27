import GradientHolder from '../GradientHolder';
import Tensor from '../Tensor';

let return_value: boolean = false;
let value: number = 0.0;

export default {
  gaussRandom() {
    if (return_value) {
      return_value = false;
      return value;
    }
    let u: number = 2 * Math.random() - 1;
    let v: number = 2 * Math.random() - 1;
    let r: number = u * u + v * v;

    if (r == 0 || r > 1) {
      return this.gaussRandom();
    }

    let c: number = Math.sqrt((-2 * Math.log(r)) / r);
    value = v * c;
    return_value = true;

    return u * c;
  },
  isArray(a: any): boolean {
    return a && a.constructor === Array;
  },
  shapeEquals(shape: any, shape2: any) {
    if (this.isArray(shape) && this.isArray(shape2)) {
      return !shape.some((v: any, k: number) => shape2[k] != v);
    } else {
      return false;
    }
  },
  shapeHardEquals(shape: any, shape2: any) {
    if (this.isArray(shape) && this.isArray(shape2)) {
      return !shape.some((v: any, k: number) => shape2[k] !== v);
    } else {
      return false;
    }
  },
  shape(array: any): number[] {
    if (!this.isArray(array)) {
      return [];
    }

    const shape: number[] = [array.length];
    let curr: any[] = array;

    while (true) {
      if (!this.isArray(curr[0])) {
        return shape;
      }

      shape.push(curr[0].length);
      curr = curr[0];
    }
  },
  buildOneDimensionalArray(
    size: number,
    buildFunc: () => any = () => 0
  ): any[] {
    const a: any[] = [];

    for (let index: number = 0; index < size; index++) {
      a.push(buildFunc());
    }

    return a;
  },
  // Random float from lower and upper bounds
  randf(lowerBound, upperBound) {
    return Math.random() * (upperBound - lowerBound) + lowerBound;
  },
  // Random int from lower and upper bounds
  randi(lowerBound, upperBound) {
    return Math.floor(this.randf(lowerBound, upperBound));
  },
  // Random number starting with mean, and standard deviation = the randomness range
  randn(mean, standardDeviation) {
    return mean + this.gaussRandom() * standardDeviation;
  },
  // Fill an array with random guassian-distributed numbers
  randnArray(w, mean, standardDeviation) {
    for (let i = 0, n = w.length; i < n; i++) {
      w[i] = this.randn(mean, standardDeviation);
    }
  },
  // Multiply two tensor matrices
  matmul(A: GradientHolder, B: GradientHolder): Tensor {
    /*const resultRowCount = A.shape[0];
    const mustEqualColCount = A.shape[1];
    const mustEqualRowCount = B.shape[0];
    const resultColCount = B.shape[1];
    // Ouput Shape: [resultRowCount, resultColCount]
    const resultTensor = new Tensor(resultRowCount, resultColCount, true);

    Assertion.assert(
      mustEqualRowCount === mustEqualColCount,
      "Matrix Multiplication Invalid Dimensions: " +
        `(${resultRowCount}, ${mustEqualColCount}), (${mustEqualRowCount}, ${resultColCount})`
    );

    // Following an algorithm: https://www.tutorialspoint.com/matrix-multiplication-algorithm
    for (let row = 0; row < resultRowCount; row++) {
      for (let col = 0; col < resultColCount; col++) {
        let dot = 0.0;

        for (let dim = 0; dim < mustEqualRowCount; dim++) {
          dot += A.get(row, dim) * B.get(dim, col);
        }

        resultTensor.set(row, col, dot);
      }
    }

    return resultTensor;*/
    return null;
  },
};
