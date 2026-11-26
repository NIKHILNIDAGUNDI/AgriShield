(function (root) {
  function applyDeepLearningFallback(document) {
    document.getElementById("dl-uninitialized-card").style.display = "block";
    document.getElementById("model-backend-selector").style.display = "none";
    document.getElementById("gpu-status").style.display = "none";
    document.getElementById("gpu-status-summary").style.display = "none";
    document.getElementById("deep-learning-section").style.display = "none";
    document.getElementById("terminal-model-name").textContent =
      "Model: RF regressor (in-browser)";
  }

  const api = { applyDeepLearningFallback };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.AgriShieldDashboardFallback = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
