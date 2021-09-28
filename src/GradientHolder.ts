export default abstract class GradientHolder {
  output: number[];
  shape: number[];
  gradv: number[];

  get(row: number, col: number, depth: number) {
    return this.output[(this.shape[1] * row + col) * this.shape[2] + depth];
  }

  set(row: number, col: number, depth: number, value: number) {
    this.output[(this.shape[1] * row + col) * this.shape[2] + depth] = value;
  }

  getGrad(row: number, col: number, depth: number) {
    return this.gradv[(this.shape[1] * row + col) * this.shape[2] + depth];
  }

  setGrad(row: number, col: number, depth: number, value: number) {
    this.gradv[(this.shape[1] * row + col) * this.shape[2] + depth] = value;
  }

  addGrad(row: number, col: number, depth: number, value: number) {
    this.gradv[(this.shape[1] * row + col) * this.shape[2] + depth] += value;
  }

  setOutput(output: number[]) {
    for (let i = 0, n = output.length; i < n; i++) {
      this.output[i] = output[i];
    }
  }

  grad(gradv: number[]): void {
    this.gradv = gradv;
  }
}
