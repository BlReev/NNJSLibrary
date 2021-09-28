export default abstract class GradientHolder {
  output: number[];
  shape: number[];
  gradv: number[];

  private ind(x: number, y: number, d: number) {
    return (this.shape[0] * y + x) * this.shape[2] + d;
  }

  get(x: number, y: number, d: number) {
    return this.output[this.ind(x, y, d)];
  }

  set(x: number, y: number, d: number, value: number) {
    this.output[this.ind(x, y, d)] = value;
  }

  getGrad(x: number, y: number, d: number) {
    return this.gradv[this.ind(x, y, d)];
  }

  setGrad(x: number, y: number, d: number, value: number) {
    this.gradv[this.ind(x, y, d)] = value;
  }

  addGrad(x: number, y: number, d: number, value: number) {
    this.gradv[this.ind(x, y, d)] += value;
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
