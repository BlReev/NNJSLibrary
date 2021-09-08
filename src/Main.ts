import ReLUPropagation from "./backpropagation/operations/ReLUPropagation";
import Library from "./utils/Library";

const tensor = Library.constant(1, 5, [1, 2, 3, -4, -5]);
const reLU = new ReLUPropagation(tensor);
reLU.grad([2, 2, 2, 2, 2]);
reLU.propagateBackwards();
console.log(reLU);
