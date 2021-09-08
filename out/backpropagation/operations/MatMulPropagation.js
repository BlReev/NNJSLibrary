"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const Utils_1 = require("../../utils/Utils");
const BinaryOperationNeuron_1 = require("../BinaryOperationNeuron");
class MatMulPropagation extends BinaryOperationNeuron_1.default {
    forwardPass() {
        return Utils_1.default.matmul(this.firstOperand, this.secondOperand);
    }
    applyFirstGradient() {
        const rows = this.firstOperand.shape[0];
        const cols = this.secondOperand.shape[1];
        const x = this.secondOperand.shape[0];
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                for (let dim = 0; dim < x; dim++) {
                    const multiplyGradient = this.gradv[cols * row + col];
                    this.firstOperand.gradv[x * row + dim] +=
                        this.secondOperand.out[cols * dim + col] * multiplyGradient;
                }
            }
        }
    }
    applySecondGradient() {
        const rows = this.firstOperand.shape[0];
        const cols = this.secondOperand.shape[1];
        const x = this.firstOperand.shape[1];
        for (let row = 0; row < rows; row++) {
            for (let col = 0; col < cols; col++) {
                for (let dim = 0; dim < x; dim++) {
                    const multiplyGradient = this.gradv[cols * row + col];
                    this.secondOperand.gradv[cols * dim + col] +=
                        this.firstOperand.out[x * row + dim] * multiplyGradient;
                }
            }
        }
    }
}
exports.default = MatMulPropagation;
//# sourceMappingURL=MatMulPropagation.js.map