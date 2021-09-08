"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const Tensor_1 = require("../../Tensor");
const UnaryOperationNeuron_1 = require("../UnaryOperationNeuron");
class ReLUPropagation extends UnaryOperationNeuron_1.default {
    forwardPass() {
        console.log(this);
        const length = this.firstOperand.out.length;
        const tensor = new Tensor_1.default(this.firstOperand.shape[0], this.firstOperand.shape[1], true);
        for (let index = 0; index < length; index++) {
            tensor.out[index] = Math.max(0, this.firstOperand.out[index]);
        }
        return tensor;
    }
    applyFirstGradient() {
        for (let index = 0; index < this.firstOperand.out.length; index++) {
            this.firstOperand.gradv[index] =
                this.firstOperand.out[index] > 0 ? this.gradv[index] : 0.0;
        }
    }
}
exports.default = ReLUPropagation;
//# sourceMappingURL=ReLUPropagation.js.map