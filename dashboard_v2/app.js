// Application Controller: Coordinates charts, tab events, early warnings, and interactive simulation
// Updated for dashboard_v2 with landing-page Executive Summary & dynamic typing AI briefing.

// Globals
let pipelineResults = null;
let currentRegion = "US";
let priceChartInstance = null;
let factorsChartInstance = null;
let lossChartInstance = null;

// Deep Learning integration globals
let dlData = null;
let activeModelBackend = "RF"; // "RF" or "Torch"

// CSS Variables for Chart styling
const chartColors = {
  US: { primary: "#00f2fe", accent: "#3b82f6", glow: "rgba(0, 242, 254, 0.25)" },
  Ukraine: { primary: "#f43f5e", accent: "#ffb199", glow: "rgba(244, 63, 94, 0.25)" },
  India: { primary: "#f59e0b", accent: "#fde047", glow: "rgba(245, 158, 11, 0.25)" },
  neutral: { border: "rgba(255, 255, 255, 0.05)", text: "#94a3b8", grid: "rgba(255, 255, 255, 0.04)" }
};

// Region default configurations for simulator
const regionDefaults = {
  US: { temp: 0.0, precip: 0, inflation: 2.5, fuel: 65.0, geopolitical_risk: 0.05, sentiment: 0.4 },
  Ukraine: { temp: 0.0, precip: 0, inflation: 5.0, fuel: 65.0, geopolitical_risk: 0.95, sentiment: -0.2 },
  India: { temp: 0.0, precip: 0, inflation: 2.4, fuel: 65.0, geopolitical_risk: 0.2, sentiment: 0.2 }
};

// Main Initialization
document.addEventListener("DOMContentLoaded", () => {
  // Hide loader with a smooth transition once models train
  setTimeout(() => {
    try {
      // Run ML pipeline in-browser
      pipelineResults = runModelingPipeline(foodSecurityDataset);
      
      // Attempt to load PyTorch Deep Learning training results
      loadDeepLearningResults();
      
      // Render Initial State
      switchRegion(currentRegion);
      renderHeatmapTable();
      initPipelineInteractions();
      
      // Populate overview cards on the summary page
      updateSummaryOverviewCards();
      
      // Hide loader
      const loader = document.getElementById("model-loader");
      loader.style.opacity = "0";
      setTimeout(() => loader.style.display = "none", 500);
    } catch (err) {
      console.error("Pipeline failure:", err);
      alert("Error initializing forecasting pipeline in browser. Please check data.js and model.js.");
    }
  }, 100);

  // Setup Event Listeners
  initEventListeners();
});

// Attempts to fetch local PyTorch training logs (if train_deep_learning.py has been run)
function loadDeepLearningResults() {
  fetch("train_results.json")
    .then(response => {
      if (!response.ok) throw new Error("JSON file missing");
      return response.json();
    })
    .then(data => {
      dlData = data;
      console.log("Deep Learning PyTorch weights loaded successfully. Active hardware:", data.gpu_model);
      
      // 1. Update Hardware & UI components
      document.getElementById("dl-uninitialized-card").style.display = "none";
      
      // Summary page GPU badge
      const gpuStatusSummary = document.getElementById("gpu-status-summary");
      gpuStatusSummary.style.display = "flex";
      document.getElementById("gpu-name-summary").textContent = `GPU: ${data.gpu_model}`;
      if (data.device.includes("cuda")) {
        gpuStatusSummary.style.background = "rgba(139, 92, 246, 0.15)";
        gpuStatusSummary.style.borderColor = "rgba(139, 92, 246, 0.3)";
      }
      
      // Detailed page GPU badge
      const gpuStatus = document.getElementById("gpu-status");
      gpuStatus.style.display = "flex";
      document.getElementById("gpu-name").textContent = `GPU: ${data.gpu_model}`;
      if (data.device.includes("cuda")) {
        gpuStatus.style.background = "rgba(139, 92, 246, 0.15)";
        gpuStatus.style.borderColor = "rgba(139, 92, 246, 0.3)";
      }
      
      document.getElementById("model-backend-selector").style.display = "flex";
      document.getElementById("deep-learning-section").style.display = "grid";
      document.getElementById("terminal-model-name").textContent = `Model: Advanced MLP (PyTorch Backend)`;
      
      // 2. Render Convergence Loss Plot
      renderLossChart(data.epoch_logs);
      
      // 3. Render Benchmarks Metrics Comparison Table
      renderBenchmarksTable(data.metrics);
      
      // 4. Force refresh of current view with backend toggles active
      switchRegion(currentRegion);
      renderHeatmapTable();
      updateSummaryOverviewCards();
    })
    .catch(err => {
      console.log("Deep Learning PyTorch results not found. Local training available via train_model.bat.");
      document.getElementById("dl-uninitialized-card").style.display = "block";
      document.getElementById("model-backend-selector").style.display = "none";
      document.getElementById("gpu-status").style.display = "none";
      document.getElementById("gpu-status-summary").style.display = "none";
      document.getElementById("deep-learning-section").style.display = "none";
      document.getElementById("terminal-model-name").textContent = `Model: RF regressor (in-browser)`;
    });
}

// Renders the epoch training curves for the neural net
function renderLossChart(epochLogs) {
  const ctxLoss = document.getElementById("lossChart").getContext("2d");
  
  if (lossChartInstance) {
    lossChartInstance.destroy();
  }

  const epochs = epochLogs.map(log => log.epoch);
  const trainLoss = epochLogs.map(log => log.train_loss);
  const valLoss = epochLogs.map(log => log.val_loss);

  lossChartInstance = new Chart(ctxLoss, {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        {
          label: "Standard Practice Errors",
          data: trainLoss,
          borderColor: "#00f2fe",
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.1
        },
        {
          label: "Test Practice Errors",
          data: valLoss,
          borderColor: "#a78bfa",
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: { color: "#94a3b8", font: { family: "Outfit", size: 10 } }
        }
      },
      scales: {
        x: {
          grid: { color: chartColors.neutral.grid },
          ticks: { color: chartColors.neutral.text, maxTicksLimit: 10, font: { size: 9 } },
          title: { display: true, text: "Practice Sessions (Epochs)", color: chartColors.neutral.text, font: { size: 9 } }
        },
        y: {
          grid: { color: chartColors.neutral.grid },
          ticks: { color: chartColors.neutral.text, font: { size: 9 } },
          title: { display: true, text: "Prediction Errors", color: chartColors.neutral.text, font: { size: 9 } }
        }
      }
    }
  });
}

// Fills comparative report card metrics
function renderBenchmarksTable(metrics) {
  document.getElementById("metric-rf-price-rmse").textContent = metrics.rf_price_rmse.toFixed(3);
  document.getElementById("metric-torch-price-rmse").textContent = metrics.torch_price_rmse.toFixed(3);
  document.getElementById("metric-rf-price-r2").textContent = `${(metrics.rf_price_r2 * 100).toFixed(1)}%`;
  document.getElementById("metric-torch-price-r2").textContent = `${(metrics.torch_price_r2 * 100).toFixed(1)}%`;
  
  document.getElementById("metric-rf-risk-rmse").textContent = metrics.rf_risk_rmse.toFixed(3);
  document.getElementById("metric-torch-risk-rmse").textContent = metrics.torch_risk_rmse.toFixed(3);
  document.getElementById("metric-rf-risk-r2").textContent = `${(metrics.rf_risk_r2 * 100).toFixed(1)}%`;
  document.getElementById("metric-torch-risk-r2").textContent = `${(metrics.torch_risk_r2 * 100).toFixed(1)}%`;

  // Price improvement badge
  const pBadge = document.getElementById("badge-price-improvement");
  pBadge.className = `improvement-badge ${metrics.price_improvement < 0 ? "negative" : ""}`;
  pBadge.innerHTML = `
    <i class="${metrics.price_improvement >= 0 ? 'fa-solid fa-square-arrow-up-right' : 'fa-solid fa-arrow-trend-down'}"></i>
    <span>${metrics.price_improvement >= 0 ? '+' : ''}${metrics.price_improvement.toFixed(1)}% Accuracy</span>
  `;

  // Risk improvement badge
  const rBadge = document.getElementById("badge-risk-improvement");
  rBadge.className = `improvement-badge ${metrics.risk_improvement < 0 ? "negative" : ""}`;
  rBadge.innerHTML = `
    <i class="${metrics.risk_improvement >= 0 ? 'fa-solid fa-square-arrow-up-right' : 'fa-solid fa-arrow-trend-down'}"></i>
    <span>${metrics.risk_improvement >= 0 ? '+' : ''}${metrics.risk_improvement.toFixed(1)}% Accuracy</span>
  `;
}

// Populates the summary cards with dynamic predictions for March 2026
function updateSummaryOverviewCards() {
  const regions = [...new Set(foodSecurityDataset.map(row => row.region))];
  regions.forEach(region => {
    let riskVal = 5.0;
    let priceVal = 110.0;
    
    if (activeModelBackend === "Torch" && dlData) {
      const match = dlData.out_of_sample_forecasts.find(p => p.region === region && p.target_month === "2026-03");
      if (match) {
        riskVal = match.torch_predicted_risk;
        priceVal = match.torch_predicted_price;
      }
    } else if (pipelineResults) {
      const regionRows = pipelineResults.laggedData.filter(row => row.region === region);
      const match = regionRows.find(row => row.year_month === "2026-03");
      if (match) {
        riskVal = match.predicted_shortage_risk;
        priceVal = match.predicted_price;
      }
    }
    
    const regionLower = region.toLowerCase();
    document.getElementById(`sum-risk-${regionLower}`).textContent = `${riskVal.toFixed(1)}%`;
    document.getElementById(`sum-price-${regionLower}`).textContent = priceVal.toFixed(1);
    
    // Dynamically update card badge classes
    const cardEl = document.getElementById(`overview-card-${regionLower}`);
    const badgeEl = cardEl.querySelector(".status-badge-mini");
    
    badgeEl.className = "status-badge-mini";
    if (riskVal < 15) {
      badgeEl.textContent = "SECURE";
      badgeEl.classList.add("risk-low");
    } else if (riskVal < 30) {
      badgeEl.textContent = "CAUTION";
      badgeEl.classList.add("risk-medium-low");
    } else if (riskVal < 50) {
      badgeEl.textContent = "WARNING";
      badgeEl.classList.add("risk-medium");
    } else {
      badgeEl.textContent = "CRITICAL";
      badgeEl.classList.add("risk-high");
    }
  });
}

// Setup Event Listeners
function initEventListeners() {
  // Page Switcher Navigation listeners
  document.getElementById("btn-show-analytics").addEventListener("click", () => {
    navigateToPage("analytics-page");
  });
  
  document.getElementById("btn-back-to-summary").addEventListener("click", () => {
    navigateToPage("summary-page");
  });

  // Landing Page Overview cards switching
  document.querySelectorAll(".overview-card").forEach(card => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".overview-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      
      currentRegion = card.dataset.region;
      
      // Update detailed view tabs to match
      document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.remove("active");
        btn.setAttribute("aria-selected", "false");
        if (btn.dataset.region === currentRegion) {
          btn.classList.add("active");
          btn.setAttribute("aria-selected", "true");
        }
      });
      
      // Synchronize dashboard region datasets under the hood
      switchRegion(currentRegion);
      
      // Reset brief terminal text to prompt selection change
      document.getElementById("briefing-text-container").innerHTML = `
        <span class="terminal-prompt">> Region selected: <strong>${currentRegion}</strong>.<br>> Ready to generate briefing. Click 'Generate Predictive Summary' below to compile intelligence.</span>
      `;
    });
  });

  // AI Briefing Terminal Generator Button
  document.getElementById("btn-generate-summary").addEventListener("click", () => {
    runAIBriefingGenerator();
  });


  // Custom Region Search Input and Button Listeners
  const fetchCustomBtn = document.getElementById("btn-fetch-custom-region");
  if (fetchCustomBtn) {
    fetchCustomBtn.addEventListener("click", () => {
      const regionName = document.getElementById("input-custom-region").value;
      fetchAndCalibrateCustomRegion(regionName);
    });
  }
  const inputCustomRegion = document.getElementById("input-custom-region");
  if (inputCustomRegion) {
    inputCustomRegion.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        const regionName = inputCustomRegion.value;
        fetchAndCalibrateCustomRegion(regionName);
      }
    });
  }

  // Detailed View Tab Switcher
  const regionSelector = document.getElementById("region-selector");
  regionSelector.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    
    document.querySelectorAll(".tab-btn").forEach(b => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    
    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    
    currentRegion = btn.dataset.region;
    
    // Sync summary page overview cards
    document.querySelectorAll(".overview-card").forEach(card => {
      card.classList.remove("active");
      if (card.dataset.region === currentRegion) card.classList.add("active");
    });

    switchRegion(currentRegion);
  });

  // Model Backend Switcher (Standard AI vs Advanced AI)
  const backendSelector = document.getElementById("model-backend-selector");
  backendSelector.addEventListener("click", (e) => {
    const btn = e.target.closest(".toggle-btn");
    if (!btn) return;
    
    document.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    
    activeModelBackend = btn.dataset.model;
    
    if (activeModelBackend === "Torch") {
      document.getElementById("terminal-model-name").textContent = "Model: Advanced MLP (PyTorch Backend)";
    } else {
      document.getElementById("terminal-model-name").textContent = "Model: RF regressor (in-browser)";
    }

    // Refresh GUI metrics with selected model forecasts
    switchRegion(currentRegion);
    renderHeatmapTable();
    updateSummaryOverviewCards();
    
    // Reset terminal prompt
    document.getElementById("briefing-text-container").innerHTML = `
      <span class="terminal-prompt">> Model backend toggled: <strong>${activeModelBackend === 'Torch' ? 'PyTorch MLP' : 'Random Forest'}</strong>.<br>> Press 'Generate Predictive Summary' to run analysis.</span>
    `;
  });

  // Simulator slider value updates
  const sliders = [
    { id: "sim-temp", valId: "val-sim-temp", suffix: "°C", plus: true },
    { id: "sim-precip", valId: "val-sim-precip", suffix: "%", plus: true },
    { id: "sim-inflation", valId: "val-sim-inflation", suffix: "%", plus: false },
    { id: "sim-fuel", valId: "val-sim-fuel", prefix: "$", suffix: "/bbl", plus: false },
    { id: "sim-geo", valId: "val-sim-geo", suffix: "", plus: false },
    { id: "sim-sentiment", valId: "val-sim-sentiment", suffix: "", plus: true }
  ];

  sliders.forEach(slider => {
    const el = document.getElementById(slider.id);
    const valEl = document.getElementById(slider.valId);
    if (!el) return;
    
    el.addEventListener("input", (e) => {
      let val = parseFloat(e.target.value);
      let text = "";
      if (slider.plus && val > 0) text += "+";
      
      if (slider.prefix) text += slider.prefix;
      text += val.toFixed(slider.id === "sim-temp" || slider.id === "sim-sentiment" ? 2 : (slider.id === "sim-inflation" ? 1 : 0));
      if (slider.suffix) text += slider.suffix;
      
      valEl.textContent = text;
      computeSimulation();
    });
  });

  // Simulator action buttons
  document.getElementById("btn-compute-sim").addEventListener("click", computeSimulation);
  document.getElementById("btn-reset-sim").addEventListener("click", resetSimulatorDefaults);
}

// Navigates between view sections smoothly
function navigateToPage(pageId) {
  const pages = ["summary-page", "analytics-page"];
  pages.forEach(id => {
    const el = document.getElementById(id);
    if (id === pageId) {
      el.style.display = "block";
      setTimeout(() => {
        el.classList.add("active");
      }, 50);
    } else {
      el.classList.remove("active");
      el.style.display = "none";
    }
  });
}

// Runs AI Report card generator in-terminal with typed HTML output
function runAIBriefingGenerator() {
  const loader = document.getElementById("briefing-loader");
  const textContainer = document.getElementById("briefing-text-container");
  
  loader.style.display = "flex";
  textContainer.innerHTML = `<span class="terminal-prompt">> Initiating predictive query for ${currentRegion.toUpperCase()}...</span>`;
  
  setTimeout(() => {
    loader.style.display = "none";
    
    const regionRows = pipelineResults.laggedData.filter(row => row.region === currentRegion);
    const lastKnown = regionRows.find(row => row.year_month === "2025-12") || regionRows[regionRows.length - 1];
    
    let m3_Price = 110.0;
    let m3_Risk = 5.0;
    
    if (activeModelBackend === "Torch" && dlData) {
      const match = dlData.out_of_sample_forecasts.find(p => p.region === currentRegion && p.target_month === "2026-03");
      if (match) {
        m3_Price = match.torch_predicted_price;
        m3_Risk = match.torch_predicted_risk;
      }
    } else if (pipelineResults) {
      const match = regionRows.find(row => row.year_month === "2026-03");
      if (match) {
        m3_Price = match.predicted_price;
        m3_Risk = match.predicted_shortage_risk;
      }
    }

    // Determine vulnerability level
    let statusText = "SECURE (LOW RISK)";
    let statusClass = "stable";
    let analysis = "";
    let action = "";
    
    if (m3_Risk > 35) {
      statusText = "CRITICAL ALERT (HIGH CONSTRAINTS)";
      statusClass = "danger";
    } else if (m3_Risk > 15) {
      statusText = "CAUTION ADVISED (ELEVATED RISK)";
      statusClass = "caution";
    }
    
    if (currentRegion === "US") {
      analysis = `The United States agricultural delivery chain remains fundamentally stable. Steady vegetation health across major crops (crop harvest density: ${lastKnown.crop_yield.toFixed(1)}%) acts as a reliable buffer, maintaining projected shortage risk at a safe level of ${m3_Risk.toFixed(1)}% through Q1 2026. However, minor price inflation is predicted: the domestic Food Price Index is expected to rise from its December actual value of ${lastKnown.food_price_index.toFixed(1)} to a forecasted ${m3_Price.toFixed(1)} in March, primarily driven by domestic shipping fuel indices and macroeconomic living costs.`;
      action = "No emergency stock release advised. Monitor transport bottlenecks and domestic diesel fuel indices for shipping shocks.";
    } else if (currentRegion === "Ukraine") {
      analysis = `Ukraine is projected to experience sustained supply tension due to persistent geopolitical friction (Tension index: ${lastKnown.geopolitical_risk.toFixed(2)}), which stands as the key risk driver. This keeps the supply line shortage risk elevated at ${m3_Risk.toFixed(1)}% through March 2026. Domestic crop indices remain heavily affected by transport costs, keeping the projected Food Price Index high at ${m3_Price.toFixed(1)} (up from the baseline of ${lastKnown.food_price_index.toFixed(1)}).`;
      action = "Establish strategic trade routes in coordination with neighboring border states. Apply freight rebates to offset transit costs.";
    } else if (currentRegion === "India") {
      analysis = `India faces severe vulnerability risks by late Q1 2026. Model inputs reveal a combination of dry seasonal precipitation anomalies and moderate harvest contraction risks. This causes the projected food price index to surge to a high of ${m3_Price.toFixed(1)} (representing a major escalation from the December baseline of ${lastKnown.food_price_index.toFixed(1)}). Concurrently, predicted supply shortage risks escalate to a critical level of ${m3_Risk.toFixed(1)}% in March 2026, marking a severe warning indicator.`;
      action = "Mobilize grain reserves pre-emptively to distribution hubs. Enforce strict shipping transport rebates and fuel offsets ahead of Q1 yield contraction.";
    } else {
      analysis = `The agricultural delivery chain for ${currentRegion} has been analyzed. Crop harvest density is estimated at ${lastKnown.crop_yield.toFixed(1)}%, with projected supply shortage risk reaching ${m3_Risk.toFixed(1)}% by March 2026. The Food Price Index is expected to follow a baseline trend, moving to a forecasted ${m3_Price.toFixed(1)} from the December actual of ${lastKnown.food_price_index.toFixed(1)}. This forecast is driven by meteorological variations, shipping costs, and regional inflation indices.`;
      action = m3_Risk > 35 
        ? "Mobilize food reserves pre-emptively to distribution hubs. Enforce strict shipping transport rebates and secure alternative import corridors."
        : (m3_Risk > 15 ? "Establish strategic trade routes in coordination with trade partners. Monitor transport bottlenecks." : "No emergency actions advised. Maintain standard monitoring of local transport and energy indices.");
    }

    textContainer.innerHTML = `
      <div class="briefing-card ${statusClass}">
        <div class="briefing-card-title">
          <i class="fa-solid fa-building-shield"></i> ${currentRegion.toUpperCase()} FORECAST BRIEF: ${statusText}
        </div>
        <div class="briefing-meta">
          <span>Predicted Price Index (Mar 26): <strong style="color: var(--accent-cyan);">${m3_Price.toFixed(1)}</strong></span>
          <span>Shortage Risk Score (Mar 26): <strong style="color: ${m3_Risk > 35 ? 'var(--accent-rose)' : (m3_Risk > 15 ? 'var(--accent-amber)' : 'var(--accent-emerald)')};">${m3_Risk.toFixed(1)}%</strong></span>
        </div>
        <p class="briefing-text" style="margin-bottom: 0.75rem; color: #cbd5e1; font-family: var(--font-body); font-size: 0.85rem; line-height: 1.6;">
          <strong>AI Security Assessment:</strong><br>
          <span id="typewriter-analysis"></span>
        </p>
        <p class="briefing-text" style="color: #ffffff; border-top: 1px dashed rgba(255, 255, 255, 0.08); padding-top: 0.5rem; font-family: var(--font-body); font-size: 0.85rem;">
          <strong>Recommended Safety Protocol:</strong><br>
          <i class="fa-solid fa-handshake-angle" style="color: var(--accent-cyan);"></i> <span id="typewriter-action"></span>
        </p>
      </div>
    `;

    // Typewrite the analysis text first, then typewrite the action text
    typeText(document.getElementById("typewriter-analysis"), analysis, 6, () => {
      typeText(document.getElementById("typewriter-action"), action, 6);
    });

  }, 900);
}

function typeText(element, text, speed, callback) {
  let idx = 0;
  element.textContent = "";
  function step() {
    if (idx < text.length) {
      element.textContent += text.charAt(idx);
      idx++;
      setTimeout(step, speed);
    } else if (callback) {
      callback();
    }
  }
  step();
}

// Switches dashboard data view for a specific region
function switchRegion(region) {
  const regionRows = pipelineResults.laggedData.filter(row => row.region === region);
  
  // Select the immediate upcoming forecast month (2026-01) as the active EWS snapshot
  let snapshot = regionRows.find(row => row.year_month === "2026-01");
  const lastKnownMonth = regionRows.find(row => row.year_month === "2025-12");
  
  if (!snapshot) snapshot = regionRows[regionRows.length - 1];
  
  let riskVal = Math.round(snapshot.predicted_shortage_risk || 5);
  let priceVal = snapshot.predicted_price;
  
  // If PyTorch backend is selected, load PyTorch values from json out-of-sample forecast
  if (activeModelBackend === "Torch" && dlData) {
    const match = dlData.out_of_sample_forecasts.find(p => p.region === region && p.target_month === "2026-01");
    if (match) {
      riskVal = Math.round(match.torch_predicted_risk);
      priceVal = match.torch_predicted_price;
    }
  }

  // Shortage risk gauge update
  document.getElementById("kpi-val-risk").textContent = `${riskVal.toFixed(1)}%`;
  document.getElementById("gauge-txt-risk").textContent = `${riskVal}%`;
  
  const gaugeFill = document.getElementById("gauge-fill-risk");
  const strokeDash = 226; // SVG circumference
  const offset = strokeDash - (riskVal / 100) * strokeDash;
  gaugeFill.style.strokeDashoffset = offset;
  
  let riskStatus = "Supply Line: Secure";
  let riskSubIcon = '<i class="fa-solid fa-shield-check"></i>';
  gaugeFill.style.stroke = "var(--accent-emerald)";
  
  if (riskVal > 65) {
    riskStatus = "Supply Line: CRITICAL THREAT";
    riskSubIcon = '<i class="fa-solid fa-triangle-exclamation" style="color: var(--accent-rose);"></i>';
    gaugeFill.style.stroke = "var(--accent-rose)";
  } else if (riskVal > 30) {
    riskStatus = "Supply Line: CAUTION REQUIRED";
    riskSubIcon = '<i class="fa-solid fa-circle-exclamation" style="color: var(--accent-amber);"></i>';
    gaugeFill.style.stroke = "var(--accent-amber)";
  }
  
  const riskSub = document.getElementById("kpi-sub-risk");
  riskSub.className = `kpi-sub ${riskVal > 65 ? "up" : (riskVal > 30 ? "warning" : "down")}`;
  riskSub.innerHTML = `${riskSubIcon} <span>${riskStatus}</span>`;
  
  // Price Index Card Update
  document.getElementById("kpi-val-price").textContent = priceVal.toFixed(1);
  
  // Calculate MoM change relative to December 2025 actual price
  const prevPrice = lastKnownMonth ? lastKnownMonth.food_price_index : 100.0;
  const momPct = ((priceVal - prevPrice) / prevPrice) * 100;
  
  const priceSub = document.getElementById("kpi-sub-price");
  priceSub.className = `kpi-sub ${momPct > 0 ? "up" : "down"}`;
  priceSub.innerHTML = `
    <i class="${momPct > 0 ? 'fa-solid fa-arrow-trend-up' : 'fa-solid fa-arrow-trend-down'}"></i>
    <span>${momPct > 0 ? '+' : ''}${momPct.toFixed(2)}% vs Dec 25 actual</span>
  `;
  
  // Crop Yield Card Update (Display December 2025 actual yield as latest available health)
  const yieldVal = lastKnownMonth ? lastKnownMonth.crop_yield.toFixed(1) : "100.0";
  document.getElementById("kpi-val-yield").textContent = `${yieldVal}%`;
  const yieldSub = document.getElementById("kpi-sub-yield");
  
  let yieldIcon = '<i class="fa-solid fa-circle-check"></i>';
  let yieldTxt = "Optimal harvest yields";
  if (parseFloat(yieldVal) < 70) {
    yieldIcon = '<i class="fa-solid fa-circle-xmark" style="color: var(--accent-rose);"></i>';
    yieldTxt = "Severe crop deficit / Logistical failure";
    yieldSub.className = "kpi-sub up";
  } else if (parseFloat(yieldVal) < 90) {
    yieldIcon = '<i class="fa-solid fa-circle-exclamation" style="color: var(--accent-amber);"></i>';
    yieldTxt = "Moderate harvest contraction observed";
    yieldSub.className = "kpi-sub warning";
  } else {
    yieldSub.className = "kpi-sub down";
  }
  yieldSub.innerHTML = `${yieldIcon} <span>${yieldTxt} (Dec 25 actual)</span>`;

  // Logistic & Cost Card Update (Display latest Dec 2025 state)
  const logCost = lastKnownMonth ? lastKnownMonth.transport_cost.toFixed(2) : "1.00";
  document.getElementById("kpi-val-logistic").textContent = logCost;
  document.getElementById("kpi-sub-logistic").innerHTML = `
    <i class="fa-solid fa-truck"></i>
    <span>Latest: Oil $${lastKnownMonth ? lastKnownMonth.fuel_price.toFixed(0) : "65"}/bbl | Inf ${lastKnownMonth ? lastKnownMonth.inflation_rate.toFixed(1) : "2.5"}%</span>
  `;

  // Refresh Charts
  updateCharts(regionRows, region);

  // Reset and update Simulator defaults to this region
  resetSimulatorDefaults();

  // Update Warning alerts center
  renderWarnings(region);
}

// Initializes and updates ChartJS instances with aligned forecast lines
function updateCharts(regionRows, region) {
  // Sort rows to guarantee correct timeline ordering
  const sortedRows = [...regionRows].sort((a, b) => a.year_month.localeCompare(b.year_month));
  
  const labels = sortedRows.map(r => r.year_month);
  const actualPrices = sortedRows.map(r => r.food_price_index); // Will be null for 2026-01 to 2026-03
  
  // Predictions: Plotted at target months (2025-01 to 2026-03)
  const forecastedPrices = sortedRows.map(r => {
    if (r.year === 2025 || r.is_out_of_sample) {
      return r.predicted_price;
    }
    return null;
  });
  
  // Speculative Spike Threshold (calculated on historical segment)
  const spikeThreshold = sortedRows.map(r => {
    return r.is_out_of_sample ? null : (r.rolling_mean_price + 1.5 * r.rolling_std_price);
  });

  const colors = chartColors[region];

  const chartDatasets = [
    {
      label: "Actual Price Index",
      data: actualPrices,
      borderColor: colors.primary,
      borderWidth: 2.5,
      pointRadius: sortedRows.map(r => r.abnormal_spike ? 5 : 0),
      pointBackgroundColor: "#ff0844",
      pointBorderColor: "#ffffff",
      pointHoverRadius: 6,
      fill: true,
      backgroundColor: "rgba(0,0,0,0)",
      tension: 0.25
    },
    {
      label: "Standard AI Forecast",
      data: forecastedPrices,
      borderColor: "#3b82f6",
      borderWidth: 2,
      borderDash: [5, 5],
      pointRadius: 4,
      pointBackgroundColor: "#3b82f6",
      pointBorderColor: "#ffffff",
      fill: false,
      tension: 0.2
    }
  ];

  // PyTorch overlay: Plotted at target months (2025-01 to 2026-03)
  if (dlData) {
    const torchForecasts = sortedRows.map(r => {
      if (r.is_out_of_sample) {
        const match = dlData.out_of_sample_forecasts.find(p => p.region === region && p.target_month === r.year_month);
        return match ? match.torch_predicted_price : null;
      } else if (r.year === 2025) {
        const match = dlData.predictions.find(p => p.region === region && p.target_month === r.year_month);
        return match ? match.torch_predicted_price : null;
      }
      return null;
    });

    chartDatasets.push({
      label: "Advanced AI Forecast",
      data: torchForecasts,
      borderColor: "#a78bfa",
      borderWidth: 2,
      pointRadius: 4,
      pointBackgroundColor: "#a78bfa",
      pointBorderColor: "#ffffff",
      fill: false,
      tension: 0.2
    });
  }

  chartDatasets.push({
    label: "Extreme Spike Threshold",
    data: spikeThreshold,
    borderColor: "rgba(245, 158, 11, 0.45)",
    borderWidth: 1.5,
    borderDash: [3, 6],
    pointRadius: 0,
    fill: false,
    tension: 0.1
  });

  if (priceChartInstance) {
    priceChartInstance.destroy();
  }

  const ctxPrice = document.getElementById("priceChart").getContext("2d");
  
  const gradientActual = ctxPrice.createLinearGradient(0, 0, 0, 350);
  gradientActual.addColorStop(0, `${colors.primary}33`);
  gradientActual.addColorStop(1, `${colors.primary}00`);
  chartDatasets[0].backgroundColor = gradientActual;

  priceChartInstance = new Chart(ctxPrice, {
    type: "line",
    data: {
      labels: labels,
      datasets: chartDatasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: { color: "#94a3b8", font: { family: "Outfit", size: 10, weight: "500" } }
        },
        tooltip: {
          backgroundColor: "rgba(8, 11, 18, 0.95)",
          titleColor: "#ffffff",
          bodyColor: "#d1d5db",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1,
          padding: 10,
          titleFont: { family: "Outfit", weight: "700" },
          bodyFont: { family: "Inter" },
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || '';
              if (label) label += ': ';
              if (context.parsed.y !== null) {
                label += context.parsed.y.toFixed(1);
              }
              
              const idx = context.dataIndex;
              if (context.datasetIndex === 0 && sortedRows[idx].abnormal_spike) {
                label += " [EXTREME PRICE SPIKE]";
              }
              return label;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: chartColors.neutral.grid },
          ticks: {
            color: chartColors.neutral.text,
            maxTicksLimit: 12,
            font: { family: "Inter", size: 10 }
          }
        },
        y: {
          grid: { color: chartColors.neutral.grid },
          ticks: {
            color: chartColors.neutral.text,
            font: { family: "Inter", size: 10 }
          }
        }
      }
    }
  });

  // B. Feature Importance Chart (Simplified text for non-technical users)
  const topFactors = pipelineResults.priceImportances.slice(0, 8);
  const factorLabels = topFactors.map(f => {
    return f.feature
      .replace("_lag_1", " (1 Month Ago)")
      .replace("_lag_2", " (2 Months Ago)")
      .replace("_lag_3", " (3 Months Ago)")
      .replace("fuel_price", "Fuel & Oil Prices")
      .replace("inflation_rate", "Living Cost Inflation")
      .replace("transport_cost", "Shipping Costs")
      .replace("geopolitical_risk", "Conflict Tension Level")
      .replace("crop_yield", "Harvest Density")
      .replace("precipitation", "Rainfall Level")
      .replace("temperature", "Temperature Deviation")
      .replace("news_sentiment", "Public News Tone")
      .replace("food_price_index", "Food Price Level")
      .replace("shortage_risk_score", "Supply Shortage Risk");
  });
  const factorValues = topFactors.map(f => f.importance * 100);

  if (factorsChartInstance) {
    factorsChartInstance.destroy();
  }

  const ctxFactors = document.getElementById("factorsChart").getContext("2d");
  const gradientBar = ctxFactors.createLinearGradient(0, 0, 300, 0);
  gradientBar.addColorStop(0, "#3b82f6aa");
  gradientBar.addColorStop(1, `${colors.primary}dd`);

  factorsChartInstance = new Chart(ctxFactors, {
    type: "bar",
    data: {
      labels: factorLabels,
      datasets: [
        {
          label: "Relative Influence (%)",
          data: factorValues,
          backgroundColor: gradientBar,
          borderColor: colors.primary,
          borderWidth: 1,
          borderRadius: 4,
          borderSkipped: "start"
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { color: chartColors.neutral.grid },
          ticks: {
            color: chartColors.neutral.text,
            font: { family: "Inter", size: 9 }
          }
        },
        y: {
          grid: { display: false },
          ticks: {
            color: "#ffffff",
            font: { family: "Outfit", size: 10, weight: "500" }
          }
        }
      }
    }
  });
}

// Populates CSS-based heat grid matching the predictions of 2025 and 2026 out-of-sample
function renderHeatmapTable() {
  const tableEl = document.getElementById("heatmap-table-el");
  const months = ["Jan 25", "Feb 25", "Mar 25", "Apr 25", "May 25", "Jun 25", "Jul 25", "Aug 25", "Sep 25", "Oct 25", "Nov 25", "Dec 25", "Jan 26", "Feb 26", "Mar 26"];
  
  let theadHTML = `<tr><th style="text-align: left; padding-left: 0.75rem;">Region</th>`;
  months.forEach(m => {
    theadHTML += `<th>${m}</th>`;
  });
  theadHTML += `</tr>`;
  tableEl.querySelector("thead").innerHTML = theadHTML;
  
  const tbody = document.getElementById("heatmap-body");
  tbody.innerHTML = "";
  
  const regions = [...new Set(foodSecurityDataset.map(row => row.region))];
  const monthLabels = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03"];
  
  regions.forEach(region => {
    const tr = document.createElement("tr");
    
    const tdLabel = document.createElement("td");
    tdLabel.className = "heatmap-region-cell";
    let flagIcon = region === "US" ? "fa-flag-usa" : (region === "Ukraine" ? "fa-circle-info" : (region === "India" ? "fa-wheat-awn" : "fa-globe"));
    tdLabel.innerHTML = `<i class="fa-solid ${flagIcon}"></i> ${region}`;
    tr.appendChild(tdLabel);
    
    monthLabels.forEach(monthStr => {
      const record = pipelineResults.laggedData.find(row => row.region === region && row.year_month === monthStr);
      
      let riskVal = record ? record.predicted_shortage_risk : 5.0;
      
      if (activeModelBackend === "Torch" && dlData) {
        if (monthStr.startsWith("2026")) {
          const match = dlData.out_of_sample_forecasts.find(p => p.region === region && p.target_month === monthStr);
          if (match) riskVal = match.torch_predicted_risk;
        } else {
          const match = dlData.predictions.find(p => p.region === region && p.target_month === monthStr);
          if (match) riskVal = match.torch_predicted_risk;
        }
      }
      
      const td = document.createElement("td");
      td.className = "heatmap-cell " + getRiskClass(riskVal);
      td.textContent = riskVal.toFixed(0);
      td.title = `${region} in ${monthStr}: Shortage Risk is ${riskVal.toFixed(1)}%`;
      
      td.addEventListener("click", () => {
        // Sync active region tabs
        document.querySelectorAll(".tab-btn").forEach(b => {
          b.classList.remove("active");
          if (b.dataset.region === region) b.classList.add("active");
        });
        currentRegion = region;
        switchRegion(region);
        
        // Scroll to simulator in analytics
        document.getElementById("simulator-panel").scrollIntoView({ behavior: "smooth" });
      });
      
      tr.appendChild(td);
    });
    
    tbody.appendChild(tr);
  });
}

// Maps numeric shortage risk score to appropriate CSS visual class
function getRiskClass(val) {
  if (val < 15) return "risk-low";
  if (val < 30) return "risk-medium-low";
  if (val < 50) return "risk-medium";
  if (val < 65) return "risk-medium-high";
  if (val < 80) return "risk-high";
  return "risk-critical";
}

// Scans all forecasted months (2025 and 2026 out-of-sample) for critical shortages (>65%)
function renderWarnings(region) {
  const container = document.getElementById("ews-alert-list-container");
  container.innerHTML = "";
  
  const regionRows = pipelineResults.laggedData.filter(row => row.region === region);
  const forecastMonths = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03"];
  
  let alerts = [];
  
  forecastMonths.forEach(monthStr => {
    const originRow = regionRows.find(r => r.year_month === monthStr);
    if (!originRow) return;
    
    let riskVal = originRow.predicted_shortage_risk;
    let priceVal = originRow.predicted_price;
    
    if (activeModelBackend === "Torch" && dlData) {
      if (monthStr.startsWith("2026")) {
        const match = dlData.out_of_sample_forecasts.find(p => p.region === region && p.target_month === monthStr);
        if (match) {
          riskVal = match.torch_predicted_risk;
          priceVal = match.torch_predicted_price;
        }
      } else {
        const match = dlData.predictions.find(p => p.region === region && p.target_month === monthStr);
        if (match) {
          riskVal = match.torch_predicted_risk;
          priceVal = match.torch_predicted_price;
        }
      }
    }
    
    if (riskVal > 65.0) {
      // Find the variables from the closest actual month (Oct, Nov, Dec 2025)
      // which predicted these out-of-sample months
      let infoRow = originRow;
      if (monthStr === "2026-01") infoRow = regionRows.find(r => r.year_month === "2025-10");
      if (monthStr === "2026-02") infoRow = regionRows.find(r => r.year_month === "2025-11");
      if (monthStr === "2026-03") infoRow = regionRows.find(r => r.year_month === "2025-12");
      if (!infoRow) infoRow = originRow;

      alerts.push({
        year_month: monthStr,
        predicted_shortage_risk: riskVal,
        predicted_price: priceVal,
        inflation_rate: infoRow.inflation_rate || 2.5,
        geopolitical_risk: infoRow.geopolitical_risk || 0.05,
        fuel_price: infoRow.fuel_price || 65.0,
        crop_yield: infoRow.crop_yield || 100.0,
        precipitation_lag_1: infoRow.precipitation_lag_1 || 50.0
      });
    }
  });
  
  const countBadge = document.getElementById("ews-alert-count");
  
  if (alerts.length === 0) {
    countBadge.textContent = "0 ACTIVE ALERTS";
    countBadge.style.color = "var(--accent-emerald)";
    container.innerHTML = `
      <div class="alert-empty">
        <i class="fa-solid fa-circle-check"></i>
        <span>No critical shortages predicted. Supply chains are safe.</span>
      </div>
    `;
    return;
  }
  
  countBadge.textContent = `${alerts.length} CRITICAL ALERTS`;
  countBadge.style.color = "var(--accent-rose)";
  
  alerts.forEach(row => {
    const alertDiv = document.createElement("div");
    alertDiv.className = "alert-item";
    
    const drivers = [];
    if (row.inflation_rate > 8.0) drivers.push("High inflation rate");
    if (row.geopolitical_risk > 0.6) drivers.push("Severe regional tensions");
    if (row.fuel_price > 85) drivers.push("High shipping & fuel costs");
    if (row.crop_yield < 80) drivers.push("Severe crop deficit");
    if (row.precipitation_lag_1 < 30) drivers.push("Severe soil drought");
    
    if (drivers.length === 0) drivers.push("Compound shipping friction");

    let policyAction = "";
    if (drivers.some(d => d.includes("drought") || d.includes("crop"))) {
      policyAction = "Mobilize food reserves from national agricultural stocks. Lift crop export limits.";
    } else if (drivers.some(d => d.includes("fuel") || d.includes("shipping"))) {
      policyAction = "Introduce diesel/fuel transportation rebates. Subsidize sea/rail shipping costs.";
    } else {
      policyAction = "Activate emergency trade corridors. Lift tariffs on agricultural imports.";
    }

    alertDiv.innerHTML = `
      <div class="alert-header">
        <span class="alert-title"><i class="fa-solid fa-triangle-exclamation"></i> SUPPLY CRISIS PREDICTED</span>
        <span class="alert-time">${row.year_month} (Forecast)</span>
      </div>
      <div class="alert-desc">
        Shortage risk rises to <strong>${row.predicted_shortage_risk.toFixed(1)}%</strong>. Projected Price Index rises to <strong>${row.predicted_price.toFixed(1)}</strong>.
      </div>
      <div class="alert-drivers">
        ${drivers.map(d => `<span class="driver-tag">${d}</span>`).join("")}
      </div>
      <div class="alert-actions">
        <strong>Recommended Safety Protocol:</strong> ${policyAction}
      </div>
    `;
    container.appendChild(alertDiv);
  });
}

// Resets sliders in simulator to default values for the selected region
function resetSimulatorDefaults() {
  const defaults = regionDefaults[currentRegion];
  if (!defaults) return;
  
  document.getElementById("sim-temp").value = defaults.temp;
  document.getElementById("val-sim-temp").textContent = (defaults.temp >= 0 ? "+" : "") + defaults.temp.toFixed(2) + "°C";
  
  document.getElementById("sim-precip").value = defaults.precip;
  document.getElementById("val-sim-precip").textContent = (defaults.precip >= 0 ? "+" : "") + defaults.precip + "%";
  
  document.getElementById("sim-inflation").value = defaults.inflation;
  document.getElementById("val-sim-inflation").textContent = defaults.inflation.toFixed(1) + "%";
  
  document.getElementById("sim-fuel").value = defaults.fuel;
  document.getElementById("val-sim-fuel").textContent = "$" + defaults.fuel.toFixed(0) + "/bbl";
  
  document.getElementById("sim-geo").value = defaults.geopolitical_risk;
  document.getElementById("val-sim-geo").textContent = defaults.geopolitical_risk.toFixed(2);
  
  document.getElementById("sim-sentiment").value = defaults.sentiment;
  document.getElementById("val-sim-sentiment").textContent = (defaults.sentiment >= 0 ? "+" : "") + defaults.sentiment.toFixed(2);

  document.querySelectorAll(".policy-card").forEach(c => {
    c.style.transform = "none";
    c.style.borderColor = "rgba(255,255,255,0.05)";
    c.style.boxShadow = "none";
  });

  computeSimulation();
}

// Runs real-time machine learning predictions based on simulator slider states
function computeSimulation() {
  const temp = parseFloat(document.getElementById("sim-temp").value);
  const precip = parseFloat(document.getElementById("sim-precip").value);
  const inflation = parseFloat(document.getElementById("sim-inflation").value);
  const fuel = parseFloat(document.getElementById("sim-fuel").value);
  const geo = parseFloat(document.getElementById("sim-geo").value);
  const sentiment = parseFloat(document.getElementById("sim-sentiment").value);

  let yieldFactor = currentRegion === "US" ? 1.0 : (currentRegion === "Ukraine" ? 0.75 : 0.9);
  let baseYield = 100 * yieldFactor;
  
  let droughtEvent = false;
  if (precip < -40 || temp > 2.5) droughtEvent = true;
  
  let cropYield = baseYield;
  if (droughtEvent) cropYield *= 0.65;
  
  const transportCost = 1.0 + 0.8 * (fuel - 50)/50 + 0.5 * geo;
  
  let computedPrice = 100.0;
  computedPrice += (100.0 - cropYield) * 1.2;
  computedPrice += (transportCost - 1.0) * 45.0;
  computedPrice += inflation * 1.5;
  computedPrice += geo * 30.0;
  computedPrice -= sentiment * 15.0;

  const droughtIndex = Math.max(0, Math.min(10, (temp + 15) / 5 + (100 - (precip + 100)/2) / 20));
  let computedRisk = (100 - cropYield) * 0.4 + (transportCost - 1.0) * 20 + geo * 25 + (10 - droughtIndex) * -2;
  computedRisk = Math.max(5.0, Math.min(95.0, computedRisk));

  const regionRows = pipelineResults.laggedData.filter(row => row.region === currentRegion && !row.is_out_of_sample);
  const lastKnownMonth = regionRows[regionRows.length - 1]; // Latest December 2025 state
  
  const featureVector = [];
  
  const simulatedState = {
    temperature: temp,
    precipitation: (precip + 100) / 2 * 10,
    crop_yield: cropYield,
    fuel_price: fuel,
    transport_cost: transportCost,
    inflation_rate: inflation,
    geopolitical_risk: geo,
    news_sentiment: sentiment,
    food_price_index: computedPrice,
    shortage_risk_score: computedRisk
  };

  pipelineResults.featureCols.forEach(col => {
    if (col.endsWith("_lag_1")) {
      const baseCol = col.replace("_lag_1", "");
      featureVector.push(simulatedState[baseCol]);
    } else if (col.endsWith("_lag_2")) {
      const baseCol = col.replace("_lag_2", "");
      featureVector.push(lastKnownMonth[`${baseCol}_lag_1`]);
    } else if (col.endsWith("_lag_3")) {
      const baseCol = col.replace("_lag_3", "");
      featureVector.push(lastKnownMonth[`${baseCol}_lag_2`]);
    }
  });

  const predictedPrice = pipelineResults.priceForest.predictRow(featureVector);
  const predictedRisk = pipelineResults.riskForest.predictRow(featureVector);

  const priceVal = document.getElementById("sim-res-price");
  const riskVal = document.getElementById("sim-res-risk");
  
  priceVal.textContent = predictedPrice.toFixed(1);
  riskVal.textContent = `${predictedRisk.toFixed(1)}%`;
  
  riskVal.className = "sim-res-val risk";
  if (predictedRisk > 65) {
    riskVal.classList.add("critical");
  } else if (predictedRisk > 30) {
    riskVal.classList.add("warning");
  }

  highlightPolicyInterventions(geo, fuel, cropYield);
}

// Highlights specific policy recommendations dynamically
function highlightPolicyInterventions(geo, fuel, cropYield) {
  const cardReserve = document.getElementById("policy-card-reserve");
  const cardSubsidize = document.getElementById("policy-card-subsidize");
  const cardTrade = document.getElementById("policy-card-trade");

  [cardReserve, cardSubsidize, cardTrade].forEach(c => {
    c.style.transform = "none";
    c.style.borderColor = "rgba(255,255,255,0.05)";
    c.style.boxShadow = "none";
  });

  if (fuel > 90) {
    cardSubsidize.style.transform = "translateY(-4px)";
    cardSubsidize.style.borderColor = "var(--accent-amber)";
    cardSubsidize.style.boxShadow = "0 4px 15px rgba(245, 158, 11, 0.15)";
  }

  if (cropYield < 75) {
    cardReserve.style.transform = "translateY(-4px)";
    cardReserve.style.borderColor = "var(--accent-blue)";
    cardReserve.style.boxShadow = "0 4px 15px rgba(59, 130, 246, 0.15)";
  }

  if (geo > 0.6) {
    cardTrade.style.transform = "translateY(-4px)";
    cardTrade.style.borderColor = "var(--accent-emerald)";
    cardTrade.style.boxShadow = "0 4px 15px rgba(16, 185, 129, 0.15)";
  }
}

// Pipeline connections interactions
function initPipelineInteractions() {
  const nodes = [
    { id: "node-meteo", desc: "Extracts monthly average temperature and rainfall totals from climate databases." },
    { id: "node-wb", desc: "Pulls official CPI consumer price inflation rates from international databases." },
    { id: "node-fallback", desc: "Collects shipping shipping freights, diesel prices, and regional risk metrics." },
    { id: "node-aggregator", desc: "Merges weather, shipping, and inflation datasets into standardized regional profiles." },
    { id: "node-lag", desc: "Calculates historical trends (comparing values from 1, 2, and 3 months ago) to guide predictions." },
    { id: "node-rf", desc: "AI Predictors. Generates 3-month advance predictions for food prices and shortage risks." }
  ];

  nodes.forEach(node => {
    const el = document.getElementById(node.id);
    if (!el) return;
    
    el.addEventListener("mouseenter", () => {
      el.classList.add("active");
      document.getElementById("system-status").innerHTML = `<i class="fa-solid fa-info-circle"></i> <span>${node.desc}</span>`;
      document.getElementById("system-status").style.color = "var(--accent-cyan)";
      document.getElementById("system-status").style.background = "rgba(0, 242, 254, 0.08)";
      document.getElementById("system-status").style.borderColor = "rgba(0, 242, 254, 0.2)";
    });

    el.addEventListener("mouseleave", () => {
      el.classList.remove("active");
      document.getElementById("system-status").innerHTML = `<div class="status-dot"></div> <span>Pipeline: Calibrated</span>`;
      document.getElementById("system-status").style.color = "var(--accent-emerald)";
      document.getElementById("system-status").style.background = "rgba(16, 185, 129, 0.08)";
      document.getElementById("system-status").style.borderColor = "rgba(16, 185, 129, 0.2)";
    });
  });
}

// Fetches, aggregates, integrates, and recalibrates forecasting models for a custom region
async function fetchAndCalibrateCustomRegion(regionName) {
  const statusDiv = document.getElementById("custom-region-status");
  const statusText = document.getElementById("custom-region-status-text");
  const fetchBtn = document.getElementById("btn-fetch-custom-region");
  const inputEl = document.getElementById("input-custom-region");
  
  if (!regionName || !regionName.trim()) {
    alert("Please enter a valid region name.");
    return;
  }
  
  // Prevent duplicate additions
  const lowerName = regionName.toLowerCase().trim();
  const existingRegions = [...new Set(foodSecurityDataset.map(row => row.region))];
  const matched = existingRegions.find(r => r.toLowerCase() === lowerName);
  if (matched) {
    alert(`Region '${matched}' is already loaded. Switching view to it.`);
    const card = document.getElementById(`overview-card-${matched.toLowerCase()}`);
    if (card) {
      card.click();
      card.scrollIntoView({ behavior: "smooth" });
    }
    return;
  }
  
  // Show status & disable controls
  statusDiv.style.display = "flex";
  statusText.textContent = `Geocoding '${regionName}'...`;
  fetchBtn.disabled = true;
  inputEl.disabled = true;
  
  try {
    // 1. Geocode the region
    const geoUrl = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(regionName)}&count=1&language=en&format=json`;
    const geoRes = await fetch(geoUrl).then(r => r.json());
    if (!geoRes.results || geoRes.results.length === 0) {
      throw new Error(`Could not resolve location '${regionName}'.`);
    }
    
    const geo = geoRes.results[0];
    const resolvedName = geo.name;
    const lat = geo.latitude;
    const lon = geo.longitude;
    const countryCode = geo.country_code || "USA";
    
    // Prevent duplicate based on resolved name
    const matchedResolved = existingRegions.find(r => r.toLowerCase() === resolvedName.toLowerCase());
    if (matchedResolved) {
      alert(`Region '${matchedResolved}' is already loaded. Switching view to it.`);
      const card = document.getElementById(`overview-card-${matchedResolved.toLowerCase()}`);
      if (card) {
        card.click();
        card.scrollIntoView({ behavior: "smooth" });
      }
      statusDiv.style.display = "none";
      fetchBtn.disabled = false;
      inputEl.disabled = false;
      return;
    }
    
    // 2. Fetch historical weather
    statusText.textContent = `Fetching climatology for lat: ${lat.toFixed(2)}, lon: ${lon.toFixed(2)}...`;
    const weatherUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}&start_date=2021-01-01&end_date=2025-12-31&daily=temperature_2m_mean,precipitation_sum&timezone=auto`;
    const weatherRes = await fetch(weatherUrl).then(r => r.json());
    const dailyWeather = weatherRes.daily || {};
    
    // 3. Fetch macroeconomic inflation rate from World Bank
    statusText.textContent = `Fetching inflation data for country code: ${countryCode}...`;
    let inflationMap = {};
    try {
      const wbUrl = `https://api.worldbank.org/v2/country/${countryCode}/indicator/FP.CPI.TOTL.ZG?date=2021:2025&format=json`;
      const wbRes = await fetch(wbUrl).then(r => r.json());
      inflationMap = parseWorldBankInflation(wbRes);
    } catch (e) {
      console.warn("World Bank inflation fetch failed, using fallback constants.", e);
    }
    
    // Show model calibrating overlay
    statusText.textContent = `Calibrating forecasting models...`;
    const modelLoader = document.getElementById("model-loader");
    modelLoader.style.display = "flex";
    modelLoader.style.opacity = "1";
    
    // Aggregate weather to monthly
    const monthlyWeather = aggregateWeatherToMonthly(dailyWeather);
    
    // 4. Generate records for 2021-01 to 2025-12 (60 months)
    const months = [];
    const dateStart = new Date("2021-01-01");
    for (let m = 0; m < 60; m++) {
      const d = new Date(dateStart);
      d.setMonth(dateStart.getMonth() + m);
      const year = d.getFullYear();
      const month = d.getMonth() + 1;
      const yearMonth = `${year}-${month.toString().padStart(2, '0')}`;
      months.push({ year, month, yearMonth });
    }
    
    const regionRecords = [];
    months.forEach(m => {
      const ym = m.yearMonth;
      const wData = monthlyWeather[ym] || {};
      
      // Determine base weather if API failed
      let temp = wData.temperature;
      let precip = wData.precipitation;
      
      const month_offset = lat < 0 ? 10 : 4;
      if (temp === null || temp === undefined) {
        temp = 20 - 12 * Math.abs(lat) / 45 + 12 * Math.sin(2 * Math.PI * (m.month - month_offset) / 12);
      }
      if (precip === null || precip === undefined) {
        precip = 55 + 25 * Math.sin(2 * Math.PI * (m.month - 3) / 12) + (Math.random() * 20 - 10);
      }
      
      // World Bank inflation or fallback
      let inflation = inflationMap[m.year];
      if (inflation === undefined || inflation === null) {
        inflation = 3.0; // standard fallback
      }
      
      // Simulate crop yield, fuel cost, price index, shortage risk
      const yieldFactor = 0.9;
      const geoRisk = 0.15;
      const newsSent = 0.2;
      
      // Simulated oil/fuel price
      const baseOil = 65 + 40 * Math.exp(-Math.pow((m.year + m.month/12.0 - 2022.4)/0.7, 2)) + (Math.random() * 6 - 3);
      const transportCost = 1.0 + 0.8 * (baseOil - 50)/50 + 0.5 * geoRisk + (Math.random() * 0.1 - 0.05);
      
      let cropYield = 100 * yieldFactor;
      if (temp > 28 && precip < 25) {
        cropYield *= 0.65; // heatwave/drought shock
      }
      cropYield += (Math.random() * 4 - 2);
      
      let priceIdx = 100.0;
      priceIdx += (100.0 - cropYield) * 1.2;
      priceIdx += (transportCost - 1.0) * 45.0;
      priceIdx += inflation * 1.5;
      priceIdx += geoRisk * 30.0;
      priceIdx -= newsSent * 15.0;
      priceIdx += (Math.random() * 8 - 4);
      
      const droughtIndex = Math.max(0, Math.min(10, (temp - 15) / 5 + (100 - precip) / 20));
      let shortageRisk = (100 - cropYield) * 0.4 + (transportCost - 1.0) * 20 + geoRisk * 25 + (10 - droughtIndex) * -2;
      shortageRisk = Math.max(5.0, Math.min(95.0, shortageRisk + (Math.random() * 4 - 2)));
      
      regionRecords.push({
        year_month: ym,
        year: m.year,
        month: m.month,
        crop_yield: cropYield,
        fuel_price: baseOil,
        transport_cost: transportCost,
        inflation_rate: inflation,
        geopolitical_risk: geoRisk,
        news_sentiment: newsSent,
        food_price_index: priceIdx,
        shortage_risk_score: shortageRisk,
        temperature: temp,
        precipitation: precip,
        region: resolvedName
      });
    });
    
    // Add to global dataset
    foodSecurityDataset.push(...regionRecords);
    
    // Recalibrate Random Forest pipeline in JavaScript
    pipelineResults = runModelingPipeline(foodSecurityDataset);
    
    // Dynamically register defaults and chart colors
    regionDefaults[resolvedName] = {
      temp: 0.0,
      precip: 0,
      inflation: 3.0,
      fuel: 65.0,
      geopolitical_risk: 0.15,
      sentiment: 0.2
    };
    chartColors[resolvedName] = {
      primary: "#a78bfa",
      accent: "#8b5cf6",
      glow: "rgba(139, 92, 246, 0.25)"
    };
    
    // Create UI Elements (Overview Card and Tab Button)
    createCustomUIElements(resolvedName);
    
    // Redraw table and update components
    renderHeatmapTable();
    updateSummaryOverviewCards();
    
    // Select the new region card to activate it
    const newCard = document.getElementById(`overview-card-${resolvedName.toLowerCase()}`);
    if (newCard) {
      newCard.click();
    }
    
    // Hide status and loading overlay
    statusDiv.style.display = "none";
    inputEl.value = "";
    
    setTimeout(() => {
      modelLoader.style.opacity = "0";
      setTimeout(() => { modelLoader.style.display = "none"; }, 500);
    }, 400);
    
  } catch (err) {
    console.error("Error in custom region forecasting:", err);
    alert(`Error: ${err.message}`);
    statusDiv.style.display = "none";
  } finally {
    fetchBtn.disabled = false;
    inputEl.disabled = false;
  }
}

// Helper to parse World Bank inflation response
function parseWorldBankInflation(res) {
  const inflationMap = {};
  if (res && res.length > 1 && Array.isArray(res[1])) {
    res[1].forEach(item => {
      const year = parseInt(item.date);
      const val = item.value;
      if (year && val !== null && val !== undefined) {
        inflationMap[year] = val;
      }
    });
  }
  return inflationMap;
}

// Helper to aggregate daily Open-Meteo weather response to monthly average temperature and total precipitation
function aggregateWeatherToMonthly(dailyData) {
  const time = dailyData.time || [];
  const temps = dailyData.temperature_2m_mean || [];
  const precips = dailyData.precipitation_sum || [];
  
  const monthly = {};
  for (let i = 0; i < time.length; i++) {
    const dateStr = time[i];
    const yearMonth = dateStr.substring(0, 7);
    if (!monthly[yearMonth]) {
      monthly[yearMonth] = { temps: [], precips: [] };
    }
    if (temps[i] !== null && temps[i] !== undefined) {
      monthly[yearMonth].temps.push(temps[i]);
    }
    if (precips[i] !== null && precips[i] !== undefined) {
      monthly[yearMonth].precips.push(precips[i]);
    }
  }
  
  const result = {};
  for (const ym in monthly) {
    const tArr = monthly[ym].temps;
    const pArr = monthly[ym].precips;
    const avgTemp = tArr.length > 0 ? tArr.reduce((s, v) => s + v, 0) / tArr.length : null;
    const sumPrecip = pArr.length > 0 ? pArr.reduce((s, v) => s + v, 0) : null;
    result[ym] = { temperature: avgTemp, precipitation: sumPrecip };
  }
  return result;
}

// Helper to create and insert overview card and tab button for a custom region
function createCustomUIElements(region) {
  const lower = region.toLowerCase();
  
  // 1. Landing Page Card
  const grid = document.querySelector(".overview-grid");
  if (!document.getElementById(`overview-card-${lower}`)) {
    const card = document.createElement("div");
    card.className = "overview-card";
    card.id = `overview-card-${lower}`;
    card.dataset.region = region;
    card.innerHTML = `
      <div class="card-border-glow"></div>
      <div class="card-main">
        <div class="card-head">
          <h3><i class="fa-solid fa-globe"></i> ${region}</h3>
          <span class="status-badge-mini risk-low">SECURE</span>
        </div>
        <div class="card-body">
          <div class="overview-metric">
            <span class="metric-lbl">Shortage Risk (Mar 26):</span>
            <span class="metric-val text-emerald" id="sum-risk-${lower}">-</span>
          </div>
          <div class="overview-metric">
            <span class="metric-lbl">Price Index (Mar 26):</span>
            <span class="metric-val text-cyan" id="sum-price-${lower}">-</span>
          </div>
        </div>
        <div class="card-foot">
          <i class="fa-solid fa-circle-nodes"></i> Custom calibrated region
        </div>
      </div>
    `;
    grid.appendChild(card);
    
    // Overview card click listener
    card.addEventListener("click", () => {
      document.querySelectorAll(".overview-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      
      currentRegion = region;
      
      // Sync tabs
      document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.remove("active");
        btn.setAttribute("aria-selected", "false");
        if (btn.dataset.region === currentRegion) {
          btn.classList.add("active");
          btn.setAttribute("aria-selected", "true");
        }
      });
      
      switchRegion(currentRegion);
      document.getElementById("briefing-text-container").innerHTML = `
        <span class="terminal-prompt">> Region selected: <strong>${currentRegion}</strong>.<br>> Ready to generate briefing. Click 'Generate Predictive Summary' below to compile intelligence.</span>
      `;
    });
  }
  
  // 2. Tabs Selector button
  const tabsContainer = document.getElementById("region-selector");
  if (!document.getElementById(`tab-${lower}`)) {
    const tabBtn = document.createElement("button");
    tabBtn.className = "tab-btn";
    tabBtn.id = `tab-${lower}`;
    tabBtn.dataset.region = region;
    tabBtn.setAttribute("role", "tab");
    tabBtn.setAttribute("aria-selected", "false");
    tabBtn.innerHTML = `<i class="fa-solid fa-globe"></i> ${region}`;
    tabsContainer.appendChild(tabBtn);
  }
}
