"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const BinaryOperationPropagation_1 = require("./BinaryOperationPropagation");
class AddPropagation extends BinaryOperationPropagation_1.default {
    forwardPass() {
        return this.firstOperand.item + this.secondOperand.item;
    }
    applyFirstGradient() {
        this.firstOperand.grad(this.gradv);
    }
    applySecondGradient() {
        this.secondOperand.grad(this.gradv);
    }
}
exports.default = AddPropagation;
//# sourceMappingURL=AddPropagation.js.map