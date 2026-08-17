/** Plain-language glossary for operator-console tooltips. */

export const GLOSSARY = {
  "phase-i": {
    label: "Phase I",
    def: "Baseline work on historical data: pick a chart, check the data, and lock control limits.",
  },
  "phase-ii": {
    label: "Phase II",
    def: "Live monitoring. New measurements are judged against the locked limits, never used to recalculate them.",
  },
  establish: {
    label: "Establish",
    def: "Run the gated Phase I pipeline on a sample so ASPC can propose chart type and limits.",
  },
  freeze: {
    label: "Freeze",
    def: "Lock the control limits to a version. After freeze, those numbers do not change when new data arrives.",
  },
  "limits-version": {
    label: "Limits version",
    def: "A unique ID for a frozen set of control limits. Phase II streams must use this exact version.",
  },
  "go-live": {
    label: "Go-live",
    def: "Attach a frozen limits version to a named stream so incoming sensor data is evaluated in real time.",
  },
  "stop-gate": {
    label: "STOP gate",
    def: "A blocking quality check. Limits will not freeze until this issue is resolved.",
  },
  gate: {
    label: "Gate",
    def: "An automated check on the data (ok, warn, or stop) before limits can freeze.",
  },
  "gage-rr": {
    label: "Gage R&R",
    def: "A measurement-system study: how much of the observed variation comes from the gage vs the parts.",
  },
  ndc: {
    label: "NDC",
    def: "Number of Distinct Categories. How many part groups the gage can tell apart. Low NDC means the gage is too coarse.",
  },
  cp: {
    label: "Cp",
    def: "Process potential: how tightly the process variation fits inside the specification, ignoring whether it is centered.",
  },
  cpk: {
    label: "Cpk",
    def: "Process capability including centering. Below 1.33 usually means the process is not capable.",
  },
  pp: {
    label: "Pp",
    def: "Overall potential using long-term variation, not just within-subgroup noise.",
  },
  ppk: {
    label: "Ppk",
    def: "Overall performance including centering, using long-term variation.",
  },
  nelson: {
    label: "Nelson",
    def: "Eight run rules that flag unusual patterns (trends, runs, hugging the center) even if no point is outside the limits.",
  },
  "western-electric": {
    label: "Western Electric",
    def: "Classic zone rules for spotting shifts using 1σ / 2σ / 3σ bands around the center line.",
  },
  wheeler: {
    label: "Wheeler",
    def: "A conservative ruleset that mainly watches points beyond the limits, reducing false alarms.",
  },
  "i-mr": {
    label: "I-MR",
    def: "Individuals and moving-range chart. One measurement at a time, plus how much consecutive points jump.",
  },
  "xbar-r": {
    label: "Xbar-R",
    def: "Subgroup average (X̄) and range (R) chart. Use when you take several parts per sample.",
  },
  checklist: {
    label: "Phase I checklist",
    def: "Ten go-live items (sample size, MSA, stability, etc.). All must pass before a stream can go live.",
  },
  timescale: {
    label: "Timescale",
    def: "The production database used for live streams. Local SQLite demos cannot register streams.",
  },
  "api-key": {
    label: "API key",
    def: "A secret the stream engine requires to register or activate a stream. Separate from your login password.",
  },
  "stream-key": {
    label: "Stream key",
    def: "The name of a live data feed, for example a line or sensor ID.",
  },
  ewma: {
    label: "EWMA",
    def: "Exponentially weighted moving average. Smooths small shifts so they show up sooner than a Shewhart chart.",
  },
} as const;
