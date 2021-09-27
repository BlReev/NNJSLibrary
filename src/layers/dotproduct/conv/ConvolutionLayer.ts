import GradientHolder from "GradientHolder";
import Tensor from "../../../Tensor";
import Assertion from "../../../utils/Assertion";
import Utils from "../../../utils/Utils";
import Layer from "../../Layer";
import OptimizableLayer from "../../OptimizableLayer";

export default class ConvolutionLayer extends OptimizableLayer {
  stride: number;
  padding: number;
  filterCount: number;
  filters: Tensor[];

  constructor(
    w: number, // Input Width
    h: number, // Input Height
    d: number, // Input Depth
    padding: number,
    filterShape: number[],
    filterCount: number,
    stride: number = null,
    bias: number = 0
  ) {
    const outputW = Math.floor((w + padding * 2 - filterShape[0]) / stride + 1);
    const outputH = Math.floor((h + padding * 2 - filterShape[1]) / stride + 1);
    const outputD = filterCount;

    super(outputW, outputH, outputD);

    this.inputShape = [w, h, d];

    this.filterCount = filterCount;
    this.filters = [];

    for (let filterIndex = 0; filterIndex < filterCount; ++filterIndex) {
      this.filters.push(
        new Tensor(filterShape[0], filterShape[1], filterShape[2])
      );
    }

    this.padding = padding;
    this.stride = stride || 1;
  }

  feedForward(inputs: Layer): Layer {
    super.feedForward(inputs);

    const stride = this.stride;

    for (let filterIndex = 0; filterIndex < this.filterCount; ++filterIndex) {
      const filter: Tensor = this.filters[filterIndex];
      let x: number = this.padding;
      let y: number = this.padding;

      for (
        let outputY = 0;
        outputY < this.outputShape[1];
        y += stride, ++outputY
      ) {
        for (
          let outputX = 0;
          outputX < this.outputShape[0];
          x += stride, ++outputX
        ) {
          let accumulator = 0;

          for (let filterY = 0; filterY < filter.shape[1]; ++filterY) {
            const inputY = y + filterY;

            for (let filterX = 0; filterX < filter.shape[0]; ++filterX) {
              const inputX = x + filterX;

              // Bounds Check
              if (
                inputY >= 0 &&
                inputY < inputs.outputShape[1] &&
                inputX >= 0 &&
                inputX < inputs.outputShape[0]
              ) {
                for (
                  let filterDepth = 0;
                  filterDepth < filter.shape[2];
                  ++filterDepth
                ) {
                  accumulator +=
                    filter.output[
                      (filter.shape[0] * filterY + filterX) * filter.shape[2] +
                        filterDepth
                    ] *
                    inputs.output[
                      (inputs.outputShape[0] * inputY + inputX) *
                        inputs.outputShape[2] +
                        filterDepth
                    ];
                }
              }
            }
          }

          accumulator += this.b.output[filterIndex];
          this.set(outputY, outputX, filterIndex, accumulator);
        }
      }
    }

    return this;
  }

  propagateBackwards() {
    const inputs = this.input;
    inputs.W.gradv = Utils.buildOneDimensionalArray(
      inputs.W.output.length,
      () => 0
    );

    const inputsW = inputs.outputShape[0];
    const inputsH = inputs.outputShape[1];
    const stride = this.stride;

    for (
      let filterIndex = 0;
      filterIndex < this.outputShape[2];
      filterIndex++
    ) {
      const filter = this.filters[filterIndex];
      let x: number;
      let y: number;
      for (
        let outputY = 0;
        outputY < this.outputShape[1];
        y += stride, outputY++
      ) {
        x = -this.padding;
        y = -this.padding;

        for (
          let outputX = 0;
          outputX < this.outputShape[0];
          x += stride, outputX++
        ) {
          const chain_grad = this.W.getGrad(outputX, outputY, filterIndex);
          for (let filterY = 0; filterY < filter.shape[1]; filterY++) {
            let inputY = y + filterY;
            for (var filterX = 0; filterX < filter.shape[0]; filterX++) {
              let inputX = x + filterX;
              if (
                inputY >= 0 &&
                inputY < inputsH &&
                inputX >= 0 &&
                inputX < inputsW
              ) {
                for (
                  let filterDepth = 0;
                  filterDepth < filter.shape[2];
                  filterDepth++
                ) {
                  const inputIndex =
                    (inputsW * inputY + inputX) * inputs.outputShape[2] +
                    filterDepth;
                  const filterIndex =
                    (filter.shape[0] * filterY + filterX) * filter.shape[2] +
                    filterDepth;

                  filter.gradv[filterIndex] +=
                    inputs.output.output[inputIndex] * chain_grad;
                  inputs.W.gradv[inputIndex] +=
                    filter.output[filterIndex] * chain_grad;
                }
              }
            }
          }
          this.b.gradv[filterIndex] += chain_grad;
        }
      }
    }
  }

  getTrainableVariables() : GradientHolder[] {
    return [
      ...super.getTrainableVariables(),
      ...this.filters
    ]
  }

  /*optimize(learningRate: number): void {
    super.optimize(learningRate);

    for (const filter of this.filters) {
      for (let index = 0; index < filter.gradv.length; index++) {
        filter.output[index] -= learningRate * filter.gradv[index];
        // console.log("filter after", filter);
      }
    }
  }*/
}
