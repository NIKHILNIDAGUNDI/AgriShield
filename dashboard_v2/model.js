// Machine Learning Module: Implements Decision Tree and Random Forest Regressors in JavaScript
// This module parses the dataset, engineers 1, 2, and 3-month lag features, trains regressors,
// and yields forecasting predictions and feature importances.
// Now updated for Dual-Stage Training to forecast out-of-sample (2026-01 to 2026-03).

class DecisionTreeRegressor {
  constructor(maxDepth = 6, minSamplesSplit = 2) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.root = null;
  }

  fit(X, y) {
    this.root = this._buildTree(X, y, 0);
  }

  _buildTree(X, y, depth) {
    const numSamples = X.length;
    if (numSamples === 0) return null;
    
    const meanVal = this._mean(y);
    
    // Leaf node conditions
    if (depth >= this.maxDepth || numSamples < this.minSamplesSplit || this._variance(y) < 1e-5) {
      return { isLeaf: true, value: meanVal };
    }

    let bestSplit = null;
    let minSSE = Infinity;
    const numFeatures = X[0].length;

    const featuresToTry = [];
    const maxFeats = Math.max(1, Math.floor(Math.sqrt(numFeatures)));
    while (featuresToTry.length < maxFeats) {
      const f = Math.floor(Math.random() * numFeatures);
      if (!featuresToTry.includes(f)) featuresToTry.push(f);
    }

    for (const f of featuresToTry) {
      const values = X.map(row => row[f]);
      const uniqueValues = [...new Set(values)].sort((a, b) => a - b);
      
      const thresholds = [];
      for (let i = 0; i < uniqueValues.length - 1; i++) {
        thresholds.push((uniqueValues[i] + uniqueValues[i+1]) / 2);
      }
      
      const stride = Math.max(1, Math.floor(thresholds.length / 15));
      for (let i = 0; i < thresholds.length; i += stride) {
        const threshold = thresholds[i];
        const { leftIdx, rightIdx } = this._split(X, f, threshold);
        
        if (leftIdx.length === 0 || rightIdx.length === 0) continue;
        
        const leftY = leftIdx.map(idx => y[idx]);
        const rightY = rightIdx.map(idx => y[idx]);
        
        const sse = this._sse(leftY) + this._sse(rightY);
        if (sse < minSSE) {
          minSSE = sse;
          bestSplit = { feature: f, threshold, leftIdx, rightIdx };
        }
      }
    }

    if (!bestSplit) {
      return { isLeaf: true, value: meanVal };
    }

    const leftX = bestSplit.leftIdx.map(idx => X[idx]);
    const leftY = bestSplit.leftIdx.map(idx => y[idx]);
    const rightX = bestSplit.rightIdx.map(idx => X[idx]);
    const rightY = bestSplit.rightIdx.map(idx => y[idx]);

    return {
      isLeaf: false,
      feature: bestSplit.feature,
      threshold: bestSplit.threshold,
      left: this._buildTree(leftX, leftY, depth + 1),
      right: this._buildTree(rightX, rightY, depth + 1)
    };
  }

  _split(X, feature, threshold) {
    const leftIdx = [];
    const rightIdx = [];
    for (let i = 0; i < X.length; i++) {
      if (X[i][feature] <= threshold) {
        leftIdx.push(i);
      } else {
        rightIdx.push(i);
      }
    }
    return { leftIdx, rightIdx };
  }

  _mean(arr) {
    return arr.reduce((sum, val) => sum + val, 0) / arr.length;
  }

  _variance(arr) {
    const m = this._mean(arr);
    return arr.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / arr.length;
  }

  _sse(arr) {
    const m = this._mean(arr);
    return arr.reduce((sum, val) => sum + Math.pow(val - m, 2), 0);
  }

  predictRow(row, node = this.root) {
    if (!node) return 0;
    if (node.isLeaf) return node.value;
    if (row[node.feature] <= node.threshold) {
      return this.predictRow(row, node.left);
    } else {
      return this.predictRow(row, node.right);
    }
  }
}

class RandomForestRegressor {
  constructor(nEstimators = 20, maxDepth = 6, minSamplesSplit = 2) {
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.trees = [];
    this.featureImportances = [];
  }

  fit(X, y) {
    this.trees = [];
    const numFeatures = X[0].length;
    
    for (let i = 0; i < this.nEstimators; i++) {
      const { sampleX, sampleY } = this._bootstrap(X, y);
      const tree = new DecisionTreeRegressor(this.maxDepth, this.minSamplesSplit);
      tree.fit(sampleX, sampleY);
      this.trees.push(tree);
    }

    this.featureImportances = new Array(numFeatures).fill(0);
    this.trees.forEach(tree => {
      this._traverseNodeImportance(tree.root, this.featureImportances);
    });
    const totalImp = this.featureImportances.reduce((s, v) => s + v, 0);
    if (totalImp > 0) {
      this.featureImportances = this.featureImportances.map(v => v / totalImp);
    } else {
      this.featureImportances = this.featureImportances.map(() => 1 / numFeatures);
    }
  }

  _traverseNodeImportance(node, importances) {
    if (!node || node.isLeaf) return;
    importances[node.feature] += 1;
    this._traverseNodeImportance(node.left, importances);
    this._traverseNodeImportance(node.right, importances);
  }

  _bootstrap(X, y) {
    const sampleX = [];
    const sampleY = [];
    const n = X.length;
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(Math.random() * n);
      sampleX.push(X[idx]);
      sampleY.push(y[idx]);
    }
    return { sampleX, sampleY };
  }

  predictRow(row) {
    const preds = this.trees.map(tree => tree.predictRow(row));
    return preds.reduce((s, v) => s + v, 0) / preds.length;
  }

  predict(X) {
    return X.map(row => this.predictRow(row));
  }
}

// Global variables to construct the model pipeline
const lagFeatures = [
  "temperature", "precipitation", "crop_yield", 
  "fuel_price", "transport_cost", "inflation_rate", 
  "geopolitical_risk", "news_sentiment", "food_price_index",
  "shortage_risk_score"
];

function prepareForecastingLags(dataset, leadMonths = 3) {
  const sorted = [...dataset].sort((a, b) => {
    if (a.region !== b.region) return a.region.localeCompare(b.region);
    return a.year_month.localeCompare(b.year_month);
  });

  const lagged = [];
  const regions = [...new Set(sorted.map(row => row.region))];
  
  regions.forEach(region => {
    const regionRows = sorted.filter(row => row.region === region);
    
    for (let i = 0; i < regionRows.length; i++) {
      const currentRow = regionRows[i];
      if (i < leadMonths) continue;
      
      const newRow = { ...currentRow };
      
      lagFeatures.forEach(col => {
        newRow[`${col}_lag_1`] = regionRows[i - 1][col];
        newRow[`${col}_lag_2`] = regionRows[i - 2][col];
        newRow[`${col}_lag_3`] = regionRows[i - leadMonths][col];
      });
      
      if (i + leadMonths < regionRows.length) {
        newRow["future_food_price_index"] = regionRows[i + leadMonths]["food_price_index"];
        newRow["future_shortage_risk_score"] = regionRows[i + leadMonths]["shortage_risk_score"];
        lagged.push(newRow);
      } else {
        newRow["future_food_price_index"] = null;
        newRow["future_shortage_risk_score"] = null;
        lagged.push(newRow);
      }
    }
  });

  return lagged;
}

function runModelingPipeline(dataset) {
  const leadMonths = 3;
  const laggedData = prepareForecastingLags(dataset, leadMonths);
  
  const featureCols = [];
  lagFeatures.forEach(col => {
    featureCols.push(`${col}_lag_1`);
    featureCols.push(`${col}_lag_2`);
    featureCols.push(`${col}_lag_3`);
  });

  // ====================================================================
  // STAGE 1: Train on year < 2025, evaluate on year === 2025 (Validation)
  // ====================================================================
  const valTrainRows = laggedData.filter(row => row.year < 2025 && row.future_food_price_index !== null);
  const valTestRows = laggedData.filter(row => row.year === 2025);
  
  const X_train_val = valTrainRows.map(row => featureCols.map(col => row[col]));
  const y_price_train_val = valTrainRows.map(row => row.future_food_price_index);
  const y_risk_train_val = valTrainRows.map(row => row.future_shortage_risk_score);
  
  const priceForestVal = new RandomForestRegressor(25, 6, 2);
  priceForestVal.fit(X_train_val, y_price_train_val);
  
  const riskForestVal = new RandomForestRegressor(25, 6, 2);
  riskForestVal.fit(X_train_val, y_risk_train_val);

  // ====================================================================
  // STAGE 2: Train on 100% of available data, predict out-of-sample
  // ====================================================================
  const prodTrainRows = laggedData.filter(row => row.future_food_price_index !== null);
  const X_train_prod = prodTrainRows.map(row => featureCols.map(col => row[col]));
  const y_price_train_prod = prodTrainRows.map(row => row.future_food_price_index);
  const y_risk_train_prod = prodTrainRows.map(row => row.future_shortage_risk_score);

  const priceForestProd = new RandomForestRegressor(25, 6, 2);
  priceForestProd.fit(X_train_prod, y_price_train_prod);

  const riskForestProd = new RandomForestRegressor(25, 6, 2);
  riskForestProd.fit(X_train_prod, y_risk_train_prod);

  // Create out-of-sample prediction rows for 2026-01, 2026-02, and 2026-03
  const finalLaggedData = [...laggedData];
  const regions = [...new Set(laggedData.map(row => row.region))];
  
  regions.forEach(region => {
    const targetMonths = ["2026-01", "2026-02", "2026-03"];
    
    targetMonths.forEach(targetMonth => {
      const outOfSampleRow = {
        year_month: targetMonth,
        year: parseInt(targetMonth.split("-")[0]),
        month: parseInt(targetMonth.split("-")[1]),
        region: region,
        crop_yield: null,
        fuel_price: null,
        transport_cost: null,
        inflation_rate: null,
        geopolitical_risk: null,
        news_sentiment: null,
        food_price_index: null,
        shortage_risk_score: null,
        temperature: null,
        precipitation: null,
        predicted_price: null,
        predicted_shortage_risk: null,
        is_out_of_sample: true
      };
      finalLaggedData.push(outOfSampleRow);
    });
  });

  // Attach predictions back to the finalLaggedData.
  // The prediction for month M represents the forecasted value for month M
  // and is computed using the features of month M - 3.
  for (let i = 0; i < finalLaggedData.length; i++) {
    const row = finalLaggedData[i];
    const regionRows = finalLaggedData.filter(r => r.region === row.region);
    regionRows.sort((a, b) => a.year_month.localeCompare(b.year_month));
    
    const rowIdx = regionRows.indexOf(row);
    if (rowIdx >= 3) {
      const featureRow = regionRows[rowIdx - 3];
      const x = featureCols.map(col => featureRow[col]);
      
      if (row.year === 2025) {
        row["predicted_price"] = priceForestVal.predictRow(x);
        row["predicted_shortage_risk"] = riskForestVal.predictRow(x);
      } else if (row.year === 2026 || row.is_out_of_sample) {
        row["predicted_price"] = priceForestProd.predictRow(x);
        row["predicted_shortage_risk"] = riskForestProd.predictRow(x);
      } else {
        // Predictions for validation / training visual check
        row["predicted_price"] = priceForestVal.predictRow(x);
        row["predicted_shortage_risk"] = riskForestVal.predictRow(x);
      }
    } else {
      row["predicted_price"] = null;
      row["predicted_shortage_risk"] = null;
    }
  }

  // Calculate rolling statistics for abnormal market spike detection
  regions.forEach(region => {
    const regionRows = finalLaggedData.filter(row => row.region === region && !row.is_out_of_sample);
    
    for (let i = 0; i < regionRows.length; i++) {
      const startIdx = Math.max(0, i - 11);
      const window = regionRows.slice(startIdx, i + 1).map(r => r.food_price_index);
      
      const mean = window.reduce((a, b) => a + b, 0) / window.length;
      const variance = window.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / window.length;
      const std = Math.sqrt(variance);
      
      regionRows[i]["rolling_mean_price"] = mean;
      regionRows[i]["rolling_std_price"] = std;
      regionRows[i]["abnormal_spike"] = regionRows[i]["food_price_index"] > (mean + 1.5 * std);
    }
  });

  // Package feature importances from validation model
  const priceImportances = featureCols.map((col, idx) => ({
    feature: col,
    importance: priceForestVal.featureImportances[idx]
  })).sort((a, b) => b.importance - a.importance);

  const riskImportances = featureCols.map((col, idx) => ({
    feature: col,
    importance: riskForestVal.featureImportances[idx]
  })).sort((a, b) => b.importance - a.importance);

  return {
    laggedData: finalLaggedData,
    priceForest: priceForestProd, // Export production forest for What-If
    riskForest: riskForestProd,
    featureCols,
    priceImportances,
    riskImportances
  };
}
