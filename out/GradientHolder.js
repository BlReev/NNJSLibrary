"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class GradientHolder {
    get(row, col) {
        return this.out[this.shape[1] * row + col];
    }
    set(row, col, value) {
        this.out[this.shape[1] * row + col] = value;
    }
    setOut(out) {
        for (let i = 0, n = out.length; i < n; i++) {
            this.out[i] = out[i];
        }
    }
    grad(gradv) {
        this.gradv = gradv;
    }
}
exports.default = GradientHolder;
//# sourceMappingURL=GradientHolder.js.map