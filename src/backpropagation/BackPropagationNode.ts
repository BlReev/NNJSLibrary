import GradientHolder from '../GradientHolder';

export default interface BackPropagationNode extends GradientHolder {
  input: GradientHolder;

  forwardPass(): GradientHolder;
  propagateBackwards(target?: number): void;
}
