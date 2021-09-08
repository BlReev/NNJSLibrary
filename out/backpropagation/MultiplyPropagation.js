"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const BinaryOperationPropagation_1 = require("./BinaryOperationPropagation");
class MultiplyPropagation extends BinaryOperationPropagation_1.default {
    forwardPass() {
        return this.firstOperand.item * this.secondOperand.item;
    }
    applyFirstGradient() {
        this.firstOperand.grad(this.secondOperand.item * this.gradv);
    }
    applySecondGradient() {
        this.secondOperand.grad(this.firstOperand.item * this.gradv);
    }
}
exports.default = MultiplyPropagation;
//# sourceMappingURL=MultiplyPropagation.js.map