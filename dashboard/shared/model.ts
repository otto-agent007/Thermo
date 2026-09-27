export type Availability = "available" | "unavailable" | "stale";
export type Member = "finite" | "equilibrium" | "frozen";
export type Horizon = 1 | 2 | 4 | 8 | 16 | 30 | "equilibrium";
export type Source = { label: string; url: string; recordedAt: string | null };
export type Status = { state: string; label: string; source: Source | null };
export type CellSummary = {
  id: string;
  seed: number;
  evaluationSeed: string;
  member: Member;
  horizon: Horizon;
  decision: string;
  sampleCount: number;
  loss: number;
  lossBounds: [number, number];
  leakage: number;
  leakageInterval: [number, number];
  survival: number;
  hopMae: number;
  asymmetryMae: number;
  sampledEvidence: string;
  exactEvidence: string;
};
export type CellDetail = CellSummary & {
  histogram: number[];
  survivalTrace: { operation: number; probability: number }[];
  identities: Record<string, string>;
  sources: Source[];
};
export type Activity = {
  availability: Availability;
  observedAt: string;
  latestFileAt: string | null;
  checkpointCount: number | null;
  expectedCheckpoints: number;
  label: string;
  issues: string[];
};
export type Milestone = {
  id: string;
  question: string;
  conclusion: string;
  state: string;
  evidenceClasses: string[];
  sources: Source[];
};
export type ResearchItem = {
  id: string;
  kind: "finding" | "deduction" | "proposal";
  title: string;
  text: string;
  scope: string;
  sources: Source[];
};
export type StudySummary = {
  counts: { fits: number; updates: number; cells: number; pairs: number };
  cells: CellSummary[];
  thresholds: {
    loss: number;
    leakage: number;
    survival: number;
    hopMae: number;
    asymmetryMae: number;
  };
  costs: {
    label: string;
    value: number | null;
    unit: string;
    scope: string;
    evidence: string;
    source: Source;
  }[];
  pairedDiagnostics: {
    label: string;
    estimate: number;
    interval: [number, number];
    conclusion: string;
  }[];
  sources: Source[];
};
export type ReportTable = {
  title: string;
  columns: string[];
  rows: string[][];
};
export type RecentStudy = {
  id: string;
  title: string;
  date: string;
  availability: Availability;
  finding: string;
  scope: string;
  gate: string;
  tables: ReportTable[];
  sources: Source[];
};
export type CyclePhase =
  | "queued"
  | "running"
  | "verifying"
  | "awaiting_review"
  | "recorded"
  | "blocked"
  | "failed";
export type ResearchCycle = {
  availability: Availability;
  observedAt: string;
  heartbeatAgeSeconds: number | null;
  notice: string | null;
  independentReview: {
    availability: "available" | "unavailable";
    notice: string | null;
    record: {
      reviewer: string;
      reviewedAt: string;
      evidenceCommit: string;
      replay: "executed" | "inspected";
      decision: "changes_requested" | "approved";
    } | null;
  } | null;
  status: {
    jobId: string;
    sourceSha: string;
    requestDigest: string;
    phase: CyclePhase;
    attempts: number;
    elapsedSeconds: number;
    heartbeatAt: string | null;
    evidenceDigest: string | null;
    reviewStatus: "pending" | "changes_requested" | "approved";
    message: string;
  } | null;
};
export type ProjectSnapshot = {
  schemaVersion: 1;
  mode: "local" | "snapshot";
  generatedAt: string;
  sourceCommit: string | null;
  availability: Availability;
  issues: string[];
  execution: Status;
  verification: Status;
  science: Status;
  study: StudySummary | null;
  recentStudies: RecentStudy[];
  cycle: ResearchCycle | null;
  activity: Activity;
  milestones: Milestone[];
  research: ResearchItem[];
};
export type Filters = {
  seeds: number[];
  members: Member[];
  horizons: Horizon[];
  query: string;
};
export type ProposalSummary = {
  id: string;
  track: "dashboard" | "research";
  objective: string;
  baseline: string;
  execution: string;
  verification: string;
  researchOutcome: string;
  review: "proposed" | "accepted" | "rejected";
  recommendation: string;
  baselineSummary: string;
  candidateSummary: string;
  patchUrl: string | null;
  reportUrl: string;
  screenshotUrls: string[];
};
export type ProposalList = { items: ProposalSummary[]; issues: string[] };
