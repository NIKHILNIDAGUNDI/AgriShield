const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const elements = {
  "model-backend-selector": { style: {} },
  "terminal-model-name": { textContent: "" },
  "api-connection-status": { style: {}, dataset: {}, textContent: "" },
};
const context = {
  AbortController: undefined,
  clearTimeout,
  console,
  document: {
    addEventListener() {},
    getElementById(id) {
      return elements[id] || null;
    },
  },
  fetch: async (url, options) => ({
    ok: true,
    json: async () => ({ status: "ok", requestedUrl: url, requestedMethod: options.method }),
  }),
  lagFeatures: ["temperature"],
  prepareForecastingLags: () => [],
  setTimeout,
};
vm.createContext(context);
vm.runInContext(
  fs.readFileSync(path.join(__dirname, "..", "app.js"), "utf8"),
  context,
  { filename: "app.js" }
);

async function main() {
  const health = await context.requestForecastApi("/v1/health");
  assert.equal(health.status, "ok");
  assert.equal(health.requestedUrl, "/v1/health");

  const apiResult = {
    api_version: "1.0.0",
    model_version: "1.0.0",
    audit: { request_hash: "request", response_hash: "response" },
    predictions: [{
      region: "Kenya",
      feature_month: "2025-12",
      target_month: "2026-03",
      predicted_price: 143.2,
      predicted_shortage_risk: 27.1,
    }],
  };
  context.applyApiPredictions(apiResult, [
    { region: "Kenya", year_month: "2025-12" },
  ]);
  assert.equal(vm.runInContext("apiPredictionActive", context), true);
  assert.equal(vm.runInContext("apiModelVersion", context), "1.0.0");
  assert.equal(vm.runInContext("activeModelBackend", context), "Torch");
  assert.equal(
    vm.runInContext("dlData.out_of_sample_forecasts[0].torch_predicted_price", context),
    143.2
  );
  assert.match(elements["terminal-model-name"].textContent, /API v1\.0\.0/);
  assert.equal(elements["model-backend-selector"].style.display, "none");

  context.fetch = async () => ({ ok: false, status: 503 });
  await assert.rejects(context.requestForecastApi("/v1/health"), /HTTP 503/);
  console.log("dashboard API client tests passed");
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
