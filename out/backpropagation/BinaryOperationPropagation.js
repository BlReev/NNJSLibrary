"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const GradientHolder_1 = require("./GradientHolder");
class BinaryOperationPropagation extends GradientHolder_1.default {
    constructor(firstOperand, secondOperand) {
        super();
        this.grad_required = true;
        this.firstOperand = firstOperand;
        this.secondOperand = secondOperand;
        this.items = this.forwardPass();
        this.out = this.items.out;
        this.grad(this.items.gradv);
    }
    propagateBackwards() {
        if (this.firstOperand.grad_required) {
            this.applyFirstGradient();
            if (this.firstOperand instanceof BinaryOperationPropagation) {
                this.firstOperand.propagateBackwards();
            }
        }
        if (this.secondOperand.grad_required) {
            this.applySecondGradient();
            if (this.secondOperand instanceof BinaryOperationPropagation)
                this.secondOperand.propagateBackwards();
        }
    }
}
exports.default = BinaryOperationPropagation;
//# sourceMappingURL=BinaryOperationPropagation.js.map