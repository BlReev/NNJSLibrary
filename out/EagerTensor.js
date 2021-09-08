"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const Tensor_1 = require("./Tensor");
class EagerTensor extends Tensor_1.default {
    constructor(n, d, inputs, grad_required = true) {
        super(n, d, grad_required);
        Object.defineProperty(this, "out", {
            value: inputs,
            writable: false,
            enumerable: true,
            configurable: true,
        });
    }
}
exports.default = EagerTensor;
//# sourceMappingURL=EagerTensor.js.map