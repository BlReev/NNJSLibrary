export default {
  assert(condition: boolean, message: string = "Value was not expected") {
    if (!condition) {
      throw new Error("AssertionError: " + message);
    }
  },
};
