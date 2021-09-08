"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const GradientHolder_1 = require("../GradientHolder");
class PropagationOperation extends GradientHolder_1.default {
    constructor(...operands) {
        super();
        this.grad_required = true;
        this.initializeOperands(...operands);
        this.items = this.forwardPass();
        this.out = this.items.out;
        this.grad(this.items.gradv);
    }
}
exports.default = PropagationOperation;
//# sourceMappingURL=PropagationOperation.js.map