import type { Milestone, ResearchItem, Source } from "../shared/model.ts";
export const archive =
  "docs/experiment-reports/2026-09-20-task-quality-inference-budget";
export const evidenceCommit = "a1b6f93052f8534f28878e92030cea7b509cae20";
export const source = (label: string, path: string): Source => ({
  label,
  url: `https://github.com/otto-agent007/Thermo/blob/${evidenceCommit}/${path}`,
  recordedAt: null,
});
export const studySource = source("Study report", `${archive}/summary.md`);
const roadmap = source("Roadmap", "docs/roadmap.md");
export const milestones: Milestone[] = [
  [
    "M1",
    "Population-objective audit",
    "One bounded equilibrium update; conditional loss improvement.",
    "Complete",
  ],
  [
    "M2",
    "Finite-sweep transfer",
    "Frozen initial/updated pairs across six finite horizons and equilibrium.",
    "Complete",
  ],
  [
    "M3",
    "Finite-sweep gradient contract",
    "Exact bounded gradient checks agree; no full-program update.",
    "Implemented",
  ],
  [
    "M4",
    "Bounded iterative refinement",
    "Five K4 updates; small gains and high leakage.",
    "Complete",
  ],
  [
    "M4B",
    "Matched training budgets",
    "No demonstrated finite-K4 advantage; primary intervals include zero.",
    "Complete",
  ],
  [
    "M4C",
    "Conservation diagnostic",
    "Uninterrupted conservation differs from terminal count-one.",
    "Complete",
  ],
  [
    "M4D",
    "Local conservation trade-off",
    "Improved local failure does not ensure full-path survival.",
    "Complete",
  ],
  [
    "M4E",
    "Context weighting",
    "Weighted arms improve matched controls, with fidelity trade-offs.",
    "Complete",
  ],
  [
    "M4F",
    "Asymmetry preservation",
    "The fixed joint preservation screen fails.",
    "Complete",
  ],
  [
    "M4G",
    "Task quality vs inference budget",
    "Neither procedure meets the joint quality contract. No savings established.",
    "Complete",
  ],
  [
    "M5",
    "Topology-aware meta-EBM",
    "Reproduce the 12-spin target, then evaluate connectivity and complete costs.",
    "Later",
  ],
].map(([id, question, conclusion, state]) => ({
  id,
  question,
  conclusion,
  state,
  evidenceClasses:
    id === "M5"
      ? []
      : ["M3", "M4D", "M4E", "M4F"].includes(id)
        ? ["exact_reference"]
        : ["software_simulation", "exact_reference"],
  sources: [id === "M4G" ? studySource : roadmap],
}));
const review = source(
  "Conservation research review",
  "docs/research/2026-09-20-conservation-directions.md",
);
export const research: ResearchItem[] = [
  {
    id: "finding",
    kind: "finding",
    title: "Conservation limits the current study",
    text: "At K30, marginal loss and local fidelity meet the contract, while terminal leakage and uninterrupted survival do not. None of the tested budgets establishes inference-sweep savings.",
    scope:
      "Recorded M4G software simulations and exact references; no physical-hardware measurement.",
    sources: [studySource],
  },
  {
    id: "local-fit",
    kind: "proposal",
    title: "Separate fitting limits from kernel limits",
    text: "Proposed next: fit the complete local target table with a larger fixed budget and predetermined starts, then evaluate survival on the same three-operation circuit.",
    scope:
      "A future diagnostic, not yet executed. The recent pilots do not prove joint infeasibility or capacity limits.",
    sources: [
      {
        label: "Full-row pilot review",
        url: "https://github.com/otto-agent007/Thermo/blob/918f67f62b9478276bc23d9e4a9a805937c58cf4/docs/experiment-reports/2026-09-23-return-fixture-full-row-pilot/review.md",
        recordedAt: "2026-09-23",
      },
    ],
  },
  {
    id: "joint",
    kind: "proposal",
    title: "Pilot a count-preserving joint sampler",
    text: "Test a separately specified pair-update law whose support preserves particle count, and account for its complete operations.",
    scope:
      "A new software execution law. It would not establish native hardware support or device savings.",
    sources: [
      review,
      {
        label: "THRML block sampling",
        url: "https://docs.thrml.ai/en/latest/api-block-sampling.html",
        recordedAt: null,
      },
    ],
  },
  {
    id: "bound",
    kind: "deduction",
    title: "A local fidelity–conservation tension",
    text: "The one-hidden-spin, uncoupled-output family constrains visible row probabilities. The row-specific leakage bound does not prove global impossibility.",
    scope:
      "Mathematical deduction requiring dedicated review before use as a research gate.",
    sources: [review],
  },
  {
    id: "hardware",
    kind: "finding",
    title: "Software capability is not hardware evidence",
    text: "Custom conditional samplers are supported in THRML. Grouping two independent Bernoulli outputs does not enforce joint cardinality.",
    scope:
      "Public APIs and papers summarized in the dated source review; no live hardware-availability claim.",
    sources: [
      review,
      {
        label: "Extropic announcement",
        url: "https://extropic.ai/writing/from-one-to-one-billion",
        recordedAt: null,
      },
    ],
  },
];
