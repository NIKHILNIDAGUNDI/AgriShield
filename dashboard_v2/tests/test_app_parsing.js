const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const dashboardDir = path.resolve(__dirname, "..");
const context = {
  document: { addEventListener() {} },
  console,
};
vm.createContext(context);
vm.runInContext(
  fs.readFileSync(path.join(dashboardDir, "app.js"), "utf8"),
  context,
  { filename: "app.js" }
);

const inflation = context.parseWorldBankInflation([
  { page: 1 },
  [
    { date: "2021", value: 4.5 },
    { date: "2025", value: -1 },
    { date: "2020", value: 8 },
    { date: "2022", value: "2.1" },
    { date: "2023", value: 101 },
    { date: "not-a-year", value: 2 },
    null,
  ],
]);
assert.deepEqual(JSON.parse(JSON.stringify(inflation)), { 2021: 4.5, 2025: -1 });
assert.deepEqual(JSON.parse(JSON.stringify(context.parseWorldBankInflation(null))), {});

const validWeatherResponse = {
  daily: {
    time: ["2024-01-01", "2024-01-02", "2024-02-01"],
    temperature_2m_mean: [10, 14, 20],
    precipitation_sum: [2, 3, 5],
  },
};
const daily = context.parseOpenMeteoDaily(validWeatherResponse);
assert.strictEqual(daily, validWeatherResponse.daily);
const monthly = context.aggregateWeatherToMonthly(daily);
assert.deepEqual(JSON.parse(JSON.stringify(monthly)), {
  "2024-01": { temperature: 12, precipitation: 5 },
  "2024-02": { temperature: 20, precipitation: 5 },
});

for (const malformed of [
  null,
  {},
  { daily: { time: [], temperature_2m_mean: [], precipitation_sum: null } },
  { daily: { time: ["2024-01-01"], temperature_2m_mean: [], precipitation_sum: [1] } },
]) {
  assert.throws(
    () => context.parseOpenMeteoDaily(malformed),
    /malformed daily observations/
  );
}

const filteredMonthly = context.aggregateWeatherToMonthly({
  time: ["2024-01-01", "2024-01-02"],
  temperature_2m_mean: [10, 101],
  precipitation_sum: [2, -1],
});
assert.deepEqual(JSON.parse(JSON.stringify(filteredMonthly)), {
  "2024-01": { temperature: 10, precipitation: 2 },
});
console.log("app API parsing tests passed");
