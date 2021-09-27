import GradientHolder from "../../GradientHolder";
import Layer from "../Layer";
import Tensor from "../../Tensor";
import Utils from "../../utils/Utils";
import OptimizableLayer from "../OptimizableLayer";
import Assertion from "../../utils/Assertion";

export default class MaxPoolingLayer extends OptimizableLayer {
  filterShape: number[];
  padding: number;
  stride: number;
  switchX: number[];
  switchY: number[];

  constructor(
    w: number,
    h: number,
    d: number,
    filterShape: number[],
    padding: number,
    stride: number = null
  ) {
    super(
      Math.floor((w + padding * 2 - filterShape[0]) / stride + 1),
      Math.floor((h + padding * 2 - filterShape[1]) / stride + 1),
      d
    );

    this.inputShape = [w, h, d];
    this.filterShape = filterShape;
    this.padding = padding;
    this.stride = stride;
    this.switchX = [];
    this.switchY = [];
  }

  feedForward(inputs: Layer): Layer {
    super.feedForward(inputs);

    this.output = new Tensor(
      this.outputShape[0],
      this.outputShape[1],
      this.outputShape[2]
    );

    let switchCounter = 0;
    for (
      let currentDepth = 0;
      currentDepth < this.outputShape[2];
      currentDepth++
    ) {
      let x = -this.padding;
      let y = -this.padding;
      for (
        let outputX = 0;
        outputX < this.outputShape[0];
        x += this.stride, outputX++
      ) {
        y = -this.padding;
        for (
          let outputY = 0;
          outputY < this.outputShape[1];
          y += this.stride, outputY++
        ) {
          let max = -Number.MIN_VALUE;
          let maxX = -1;
          let maxY = -1;

          for (let filterX = 0; filterX < this.filterShape[0]; filterX++) {
            for (let filterY = 0; filterY < this.filterShape[1]; filterY++) {
              const originalY = y + filterY;
              const originalX = x + filterX;

              if (
                originalY >= 0 &&
                originalY < inputs.outputShape[1] &&
                originalX >= 0 &&
                originalX < inputs.outputShape[0]
              ) {
                const currentValue = inputs.get(
                  originalX,
                  originalY,
                  currentDepth
                );

                if (currentValue > max) {
                  max = currentValue;
                  maxX = originalX;
                  maxY = originalY;
                }
              }
            }
          }
          this.switchX[switchCounter] = maxX;
          this.switchY[switchCounter] = maxY;
          switchCounter++;
          this.output.set(outputY, outputX, currentDepth, max);
        }
      }
    }

    return this;
  }

  propagateBackwards(): void {
    const inputs: Layer = this.input;

    inputs.output.gradv = Utils.buildOneDimensionalArray(
      inputs.W.output.length,
      () => 0
    );
    let output = this.output;

    let switchIndex = 0;
    for (
      let currentDepth = 0;
      currentDepth < this.outputShape[2];
      currentDepth++
    ) {
      let x = -this.padding;
      let y = -this.padding;
      for (
        let outputX = 0;
        outputX < this.outputShape[0];
        x += this.stride, outputX++
      ) {
        y = -this.padding;
        for (
          let outputY = 0;
          outputY < this.outputShape[1];
          y += this.stride, outputY++
        ) {
          let chain_grad = this.W.getGrad(outputX, outputY, currentDepth);
          inputs.W.setGrad(
            this.switchY[switchIndex],
            this.switchX[switchIndex],
            currentDepth,
            chain_grad
          );
          switchIndex++;
        }
      }
    }
  }
}
