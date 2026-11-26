# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary users are emergency food-security analysts who need to identify regional price pressure and shortage risk quickly.

## Product Purpose

AgriShield provides regional food-price and shortage-risk forecasting with operational early-warning indicators, model comparison, scenario simulation, and custom-region analysis.

## Positioning

The product combines historical climate and macroeconomic signals with three-month forecasts, risk alerts, and auditable model metadata to support timely food-security intervention decisions.

## Operating Context

Analysts use the dashboard for regional monitoring, model comparison, threshold alerts, what-if analysis, and custom-region calibration. The interface should remain usable by keyboard and on desktop or mobile.

## Capabilities and Constraints

- Preserve regional forecasts, model comparison, charts, custom-region lookup, alerts, and provenance labels.
- The existing Python API exposes `/v1/health`, `/v1/models`, `/v1/predict`, `/v1/audit`, and `/v1/monitoring`.
- Synthetic, estimated, and observed fields must remain explicitly distinguished.
- The React redesign must compile with TypeScript and remain verifiable through Playwright.

## Brand Commitments

- Existing product name: AgriShield.
- Focus on operational clarity, early-warning use, and evidence-backed forecasting.
- No invented real-world benchmarks or customer claims.

## Evidence on Hand

- Existing dashboard in `dashboard_v2/index.html`, `app.js`, `model.js`, and `data.js`.
- Python model, artifact registry, API, tests, and deployment configuration in the repository.

## Product Principles

1. Make risk and uncertainty immediately legible.
2. Keep model evidence adjacent to every decision surface.
3. Preserve static fallback behavior when the API is unavailable.
4. Distinguish observed, estimated, and synthetic data.

## Accessibility & Inclusion

Maintain keyboard navigation, visible focus, readable contrast, reduced-motion support, semantic controls, and responsive layouts.
