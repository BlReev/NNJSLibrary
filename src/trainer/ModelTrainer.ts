import GradientHolder from "../GradientHolder";
import Tensor from "../Tensor";
import Assertion from "../utils/Assertion";
import Utils from "../utils/Utils";
import Model from "../model/Model";

interface TrainResult {
  forwardTime: number;
  backwardTime: number;
  costLoss: number;
  loss: number;
  output: number[];
}

export default class ModelTrainer {
  private model: Model;
  private learningRate: number;
  private trainBatchSize: number;
  private currentBatchSize: number;

  constructor(model: Model, learningRate: number, trainBatchSize: number) {
    this.model = model;
    this.learningRate = learningRate;
    this.trainBatchSize = trainBatchSize;
    this.currentBatchSize = 0;
  }

  trainStep(inputs: GradientHolder, targetLabel: number): TrainResult {
    const startForward = Date.now();
    const output: number[] = this.model.guess(inputs).output;
    const forwardTime = Date.now() - startForward;
    const startBackward = Date.now();
    const costLoss: number = this.model.backward(targetLabel);
    const backwardTime = Date.now() - startBackward;

    if (this.currentBatchSize++ % this.trainBatchSize === 0) {
      this.model.optimize(this.trainBatchSize);
    }

    return {
      forwardTime,
      backwardTime,
      costLoss,
      loss: costLoss /* TODO: Implement Weight Decay Loss */,
      output,
    };
  }

  trainEpoch(inputs: GradientHolder[] | number[][], labels: number[]) {
    if (inputs.length > 0) {
      Assertion.assert(
        inputs.length === labels.length,
        `Inputs and Labels length's do not match!`
      );
      const sets = inputs.length;
      const inputShape = this.model.inputShape;
      const numInputs = inputShape[0] * inputShape[1] * inputShape[2];
      const tensorInputs: GradientHolder[] = [];

      for (let inputIndex = 0; inputIndex < inputs.length; ++inputIndex) {
        const input = inputs[inputIndex];
        if (input instanceof Array) {
          const inputTensor = new Tensor(
            inputShape[0],
            inputShape[1],
            inputShape[2]
          );
          inputTensor.output = input;

          Assertion.assert(
            inputTensor.output.length === numInputs,
            `Input Shape Must Equal Model's Input Shape\n  Input Index: ${inputIndex}\n  Shape Expected: ${inputShape}`
          );

          tensorInputs[inputIndex] = inputTensor;
          inputs[inputIndex] = undefined;
        } else {
          Assertion.assert(
            Utils.shapeEquals(input.shape, inputShape),
            `Input Shape Must Equal Model's Input Shape\n  Input Index: ${inputIndex}\n  Shape Expected: ${inputShape}`
          );

          tensorInputs[inputIndex] = input;
        }
      }

      const startEpoch = Date.now();
      let sumStepTime = 0;
      for (let setIndex = 0; setIndex < sets; ++setIndex) {
        const trainResult: TrainResult = this.trainStep(
          tensorInputs[setIndex],
          labels[setIndex]
        );
        sumStepTime += trainResult.forwardTime + trainResult.backwardTime;

        if (this.currentBatchSize % this.trainBatchSize === 0) {
          const set = setIndex + 1;
          const doneRatio = set / sets;
          const finishSymbol = "=";
          const unfinishSymbol = " ";
          const symbolCount = 25;
          const finishCount = Math.round(symbolCount * doneRatio);
          const unfinishCount = symbolCount - finishCount;
          process.stdout.write(
            `${set}/${sets} [${
              finishSymbol.repeat(finishCount) +
              unfinishSymbol.repeat(unfinishCount)
            }] - ${Utils.timeString(Date.now() - startEpoch)} ${Math.round(
              sumStepTime / set
            )}ms/step - loss: ${trainResult.loss}` +
              (set !== sets ? "\r" : "\n")
          );
        }
      }
    }
  }
}
