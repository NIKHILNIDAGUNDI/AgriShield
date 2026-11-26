const assert = require("node:assert/strict");
const { applyDeepLearningFallback } = require("../dashboard_fallback.js");

const ids = [
  "dl-uninitialized-card",
  "model-backend-selector",
  "gpu-status",
  "gpu-status-summary",
  "deep-learning-section",
  "terminal-model-name",
];
const elements = Object.fromEntries(ids.map(id => [
  id,
  { style: { display: "flex" }, textContent: "PyTorch model" },
]));
const document = {
  addEventListener() {},
  getElementById(id) {
    assert.ok(elements[id], `unexpected element lookup: ${id}`);
    return elements[id];
  },
};

function testBrowserModelFallback() {
  applyDeepLearningFallback(document);

  assert.equal(elements["dl-uninitialized-card"].style.display, "block");
  for (const id of [
    "model-backend-selector",
    "gpu-status",
    "gpu-status-summary",
    "deep-learning-section",
  ]) {
    assert.equal(elements[id].style.display, "none");
  }
  assert.equal(
    elements["terminal-model-name"].textContent,
    "Model: RF regressor (in-browser)"
  );
  console.log("dashboard fallback tests passed");
}

testBrowserModelFallback();
