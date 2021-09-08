"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const PropagationOperation_1 = require("./PropagationOperation");
class UnaryOperationNeuron extends PropagationOperation_1.default {
    constructor(firstOperand) {
        super(firstOperand);
    }
    initializeOperands(...operands) {
        this.firstOperand = operands[0];
    }
    propagateBackwards() {
        if (this.firstOperand.grad_required) {
            this.applyFirstGradient();
            if (this.firstOperand instanceof PropagationOperation_1.default) {
                this.firstOperand.propagateBackwards();
            }
        }
    }
}
exports.default = UnaryOperationNeuron;
//# sourceMappingURL=UnaryOperationNeuron.js.map