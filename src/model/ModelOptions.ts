import LossType from '../loss/LossType';

export default interface ModelOptions {
  optimizer?: string;
  loss?: string;
  learningRate: number;
}
