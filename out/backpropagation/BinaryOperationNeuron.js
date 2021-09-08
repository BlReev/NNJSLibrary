"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const PropagationOperation_1 = require("./PropagationOperation");
class BinaryOperationNeuron extends PropagationOperation_1.default {
    constructor(firstOperand, secondOperand) {
        super(firstOperand, secondOperand);
    }
    initializeOperands(...operands) {
        this.firstOperand = operands[0];
        this.secondOperand = operands[1];
    }
    propagateBackwards() {
        if (this.firstOperand.grad_required) {
            this.applyFirstGradient();
            if (this.firstOperand instanceof PropagationOperation_1.default) {
                this.firstOperand.propagateBackwards();
            }
        }
        if (this.secondOperand.grad_required) {
            this.applySecondGradient();
            if (this.secondOperand instanceof PropagationOperation_1.default)
                this.secondOperand.propagateBackwards();
        }
    }
}
exports.default = BinaryOperationNeuron;
//# sourceMappingURL=BinaryOperationNeuron.js.map