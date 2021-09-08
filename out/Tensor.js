"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const GradientHolder_1 = require("./GradientHolder");
const Utils_1 = require("./utils/Utils");
class Tensor extends GradientHolder_1.default {
    constructor(n, d, grad_required = true) {
        super();
        this.shape = [n, d];
        this.out = Utils_1.default.buildOneDimensionalArray(n * d, () => 0);
        this.grad(Utils_1.default.buildOneDimensionalArray(n * d, () => 0));
        this.grad_required = grad_required;
    }
    fillGaussianRandom(mean, standardDeviation) {
        Utils_1.default.randnArray(this.out, mean, standardDeviation);
        return this;
    }
}
exports.default = Tensor;
//# sourceMappingURL=Tensor.js.map