export default abstract class GradientHolder {
  out: number[];
  shape: number[];
  gradv: number[];
  grad_required: boolean;

  get(row: number, col: number) {
    return this.out[this.shape[1] * row + col];
  }

  set(row: number, col: number, value: number) {
    this.out[this.shape[1] * row + col] = value;
  }

  setOut(out: number[]) {
    for (let i = 0, n = out.length; i < n; i++) {
      this.out[i] = out[i];
    }
  }

  grad(gradv: number[]): void {
    this.gradv = gradv;
  }
}
