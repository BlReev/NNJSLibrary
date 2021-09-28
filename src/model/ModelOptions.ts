import Optimizer from "../optimizer/Optimizer";

export default interface ModelOptions {
  optimizer?: string | Optimizer;
  loss?: string;
  learningRate: number;
}
