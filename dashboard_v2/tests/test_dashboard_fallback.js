const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { applyDeepLearningFallback } = require("../dashboard_fallback.js");
const vm = require("node:vm");

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

const dashboardDir = path.resolve(__dirname, "..");
const context = {
  document,
  console: { log() {}, error() {} },
  fetch: () => Promise.reject(new Error("training results unavailable")),
  AgriShieldDashboardFallback: { applyDeepLearningFallback },
};
vm.createContext(context);
vm.runInContext(
  fs.readFileSync(path.join(dashboardDir, "app.js"), "utf8"),
  context,
  { filename: "app.js" }
);

async function testFailedResultsFetchFallsBackToBrowserModel() {
  context.loadDeepLearningResults();
  await new Promise(resolve => setImmediate(resolve));

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
  console.log("dashboard fallback integration tests passed");
}

testFailedResultsFetchFallsBackToBrowserModel().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
