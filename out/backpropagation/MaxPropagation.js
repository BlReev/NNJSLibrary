"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const BinaryOperationPropagation_1 = require("./BinaryOperationPropagation");
class MaxPropagation extends BinaryOperationPropagation_1.default {
    forwardPass() {
        return Math.max(this.firstOperand.item, this.secondOperand.item);
    }
    applyFirstGradient() {
        this.firstOperand.grad((this.firstOperand.item >= this.secondOperand.item ? 1 : 0) * this.gradv);
    }
    applySecondGradient() {
        this.secondOperand.grad((this.firstOperand.item <= this.secondOperand.item ? 1 : 0) * this.gradv);
    }
}
exports.default = MaxPropagation;
//# sourceMappingURL=MaxPropagation.js.map