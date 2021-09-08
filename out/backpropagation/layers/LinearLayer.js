"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const Tensor_1 = require("../../Tensor");
const Assertion_1 = require("../../utils/Assertion");
const BinaryOperationNeuron_1 = require("../BinaryOperationNeuron");
const GradientHolder_1 = require("../../GradientHolder");
const AddPropagation_1 = require("../operations/AddPropagation");
const MatMulPropagation_1 = require("../operations/MatMulPropagation");
class LinearLayout extends GradientHolder_1.default {
    constructor(inputDimension, outputDimension) {
        super();
        this.shape = [inputDimension, outputDimension];
        this.W = new Tensor_1.default(inputDimension, outputDimension).fillGaussianRandom(0, 0.88);
        this.b = new Tensor_1.default(outputDimension, 1).fillGaussianRandom(0, 0.88);
    }
    forwardPass() {
        return this.items;
    }
    propagateBackwards() {
        this.items.grad(this.gradv);
        if (this.items instanceof BinaryOperationNeuron_1.default) {
            this.items.propagateBackwards();
        }
    }
    feedForward(inputs) {
        Assertion_1.default.assert(inputs.out.length === this.shape[0], "Feed Forward was given an invalid input count.");
        const matmulOperation = new MatMulPropagation_1.default(inputs, this.W);
        this.items = new AddPropagation_1.default(matmulOperation, this.b);
        this.shape = matmulOperation.shape;
        this.out = this.items.out;
        this.grad(this.items.gradv);
        return this;
    }
}
exports.default = LinearLayout;
//# sourceMappingURL=LinearLayer.js.map