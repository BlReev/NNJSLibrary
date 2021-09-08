import GradientHolder from "../GradientHolder";

export default interface BackPropagationNode extends GradientHolder {
  items: GradientHolder;

  forwardPass(): GradientHolder;
  propagateBackwards(): void;
}
