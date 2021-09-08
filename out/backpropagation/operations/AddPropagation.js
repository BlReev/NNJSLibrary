"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const Tensor_1 = require("../../Tensor");
const BinaryOperationNeuron_1 = require("../BinaryOperationNeuron");
class AddPropagation extends BinaryOperationNeuron_1.default {
    forwardPass() {
        const length = this.firstOperand.out.length;
        const tensor = new Tensor_1.default(1, length, true);
        for (let i = 0; i < length; i++) {
            tensor.out[i] = this.firstOperand.out[i] + this.secondOperand.out[i];
        }
        return tensor;
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