"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const EagerTensor_1 = require("../EagerTensor");
const Tensor_1 = require("../Tensor");
const Assertion_1 = require("./Assertion");
exports.default = {
    // Returns a tensor which output can be changed
    variable(n, d, inputs) {
        Assertion_1.default.assert(n * d === inputs.length, "Input length does not match the shape");
        const tensor = new Tensor_1.default(n, d);
        tensor.setOut(inputs);
        return tensor;
    },
    // Returns a constant tensor which output can't be changed
    constant(n, d, inputs) {
        Assertion_1.default.assert(n * d === inputs.length, "Input length does not match the shape");
        return new EagerTensor_1.default(n, d, inputs);
    },
};
//# sourceMappingURL=Library.js.map