"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = {
    assert(condition, message = "Value was not expected") {
        if (!condition) {
            throw new Error("AssertionError: " + message);
        }
    },
};
//# sourceMappingURL=Assertion.js.map