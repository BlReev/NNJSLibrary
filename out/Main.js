"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const ReLUPropagation_1 = require("./backpropagation/operations/ReLUPropagation");
const Library_1 = require("./utils/Library");
const tensor = Library_1.default.constant(1, 5, [1, 2, 3, -4, -5]);
const reLU = new ReLUPropagation_1.default(tensor);
reLU.grad([2, 2, 2, 2, 2]);
reLU.propagateBackwards();
console.log(reLU);
//# sourceMappingURL=Main.js.map