import { parse } from "lossless-json";
import { createHash } from "node:crypto";
import { z } from "zod";
import pins from "./archive-pins.json" with { type: "json" };
import {
  archive,
  evidenceCommit,
  milestones,
  research,
  source,
  studySource,
} from "./catalog.ts";
import { readBounded } from "./files.ts";
import { loadRecentStudies } from "./recent.ts";
import type {
  CellDetail,
  CellSummary,
  ProjectSnapshot,
  StudySummary,
} from "../shared/model.ts";
export function parseEvidence(text: string): unknown {
  return parse(text, undefined, {
    parseNumber: (token: string) =>
      /^-?\d+$/.test(token) && !Number.isSafeInteger(Number(token))
        ? token
        : Number(token),
  });
}
const num = z.number().finite();
const interval = z.tuple([num, num]);
const cellSchema = z.object({
  cell_id: z.string(),
  seed: z.number().int(),
  evaluation_seed: z.union([z.string(), z.number()]),
  member: z.enum(["finite", "equilibrium", "frozen"]),
  horizon: z.union([
    z.literal(1),
    z.literal(2),
    z.literal(4),
    z.literal(8),
    z.literal(16),
    z.literal(30),
    z.literal("equilibrium"),
  ]),
  sample_count: z.number().positive(),
  leakage_count: num,
  population_loss_estimate: num,
  evidence_class: z.string(),
  parameter_digest: z.string(),
  tables_digest: z.string(),
  particle_histogram: z.array(z.number().int().nonnegative()).length(26),
  decision: z.object({
    status: z.string(),
    loss_bounds: interval,
    leakage_interval: interval,
  }),
  exact_metrics: z.object({
    evidence_class: z.string(),
    hop_mae: num,
    asymmetry_mae: num,
    survival: z
      .array(
        z.object({
          operation: z.number().int(),
          survival_probability: num.min(0).max(1),
        }),
      )
      .length(500),
  }),
});
const countSchema = z.object({
  endpoint_draws: num,
  modeled_sweeps: num.nullable(),
});
const studySchema = z.object({
  schema_version: z.literal("quality_budget_complete_study.v1"),
  result_digest: z.string(),
  request_hash: z.string(),
  fits: z.array(z.object({ steps: z.array(z.unknown()) })),
  evaluations: z.array(
    z.object({
      seed: z.number(),
      cells: z.array(cellSchema),
      pairs: z.array(
        z.object({
          statistics: z.object({
            population_objective_difference_after_minus_before: num,
            paired_jackknife_normal_95_interval: interval,
            population_objective_conclusion: z.string(),
          }),
        }),
      ),
    }),
  ),
  request: z.object({
    protocol: z.object({
      loss_max: num,
      leakage_max: num,
      survival_min: num,
      hop_mae_max: num,
      asymmetry_mae_max: num,
    }),
  }),
  costs: z.object({
    evidence_class: z.string(),
    training: z.object({
      finite: countSchema,
      equilibrium: countSchema,
      endpoint_draws: num,
    }),
    evaluation: z.object({ endpoint_draws: num }),
  }),
});
export async function loadEvidence(
  repoRoot: string,
): Promise<{ snapshot: ProjectSnapshot; details: Map<string, CellDetail> }> {
  const now = new Date().toISOString();
  const snapshot: ProjectSnapshot = {
    schemaVersion: 1,
    mode: "local",
    generatedAt: now,
    sourceCommit: evidenceCommit,
    availability: "unavailable",
    issues: [],
    execution: { state: "unknown", label: "Unavailable", source: studySource },
    verification: {
      state: "unknown",
      label: "Unavailable",
      source: studySource,
    },
    science: { state: "unknown", label: "Unavailable", source: studySource },
    study: null,
    recentStudies: await loadRecentStudies(repoRoot),
    activity: {
      availability: "unavailable",
      observedAt: now,
      latestFileAt: null,
      checkpointCount: null,
      expectedCheckpoints: 21,
      label: "Local activity unavailable",
      issues: [],
    },
    milestones,
    research,
  };
  const details = new Map<string, CellDetail>();
  try {
    // Byte pins authenticate the approved complete bundle without numerical replay.
    const texts: Record<string, string> = {};
    for (const [name, expected] of Object.entries(pins)) {
      const { text } = await readBounded(repoRoot, `${archive}/${name}`);
      if (createHash("sha256").update(text).digest("hex") !== expected)
        throw new Error(
          `${name}: changed or unsupported archive; recorded verification withheld`,
        );
      texts[name] = text;
    }
    const s = studySchema.parse(parseEvidence(texts["study.json"]));
    const completion = z
      .object({
        full_m4g_complete: z.literal(true),
        result_digest: z.string(),
        scientific_decision: z.literal("both_quality_failure"),
      })
      .parse(parseEvidence(texts["completion.json"]));
    if (completion.result_digest !== s.result_digest)
      throw new Error("Completion identity mismatch");
    const sources = [
      studySource,
      source(
        "Frozen protocol",
        "docs/experiments/task-quality-inference-budget.md",
      ),
      source("Recorded verification", `${archive}/verification.md`),
      source("Independent reviews", `${archive}/evidence-review.json`),
      source("Full recorded CSV", `${archive}/metrics.csv`),
      source("Raw study", `${archive}/study.json`),
    ];
    const cells: CellSummary[] = s.evaluations
      .flatMap((e) => e.cells)
      .map((c) => {
        const row: CellSummary = {
          id: c.cell_id,
          seed: c.seed,
          evaluationSeed: String(c.evaluation_seed),
          member: c.member,
          horizon: c.horizon,
          decision: c.decision.status,
          sampleCount: c.sample_count,
          loss: c.population_loss_estimate,
          lossBounds: c.decision.loss_bounds,
          leakage: c.leakage_count / c.sample_count,
          leakageInterval: c.decision.leakage_interval,
          survival: c.exact_metrics.survival.at(-1)!.survival_probability,
          hopMae: c.exact_metrics.hop_mae,
          asymmetryMae: c.exact_metrics.asymmetry_mae,
          sampledEvidence: c.evidence_class,
          exactEvidence: c.exact_metrics.evidence_class,
        };
        details.set(row.id, {
          ...row,
          histogram: c.particle_histogram,
          survivalTrace: c.exact_metrics.survival.map((x) => ({
            operation: x.operation,
            probability: x.survival_probability,
          })),
          identities: {
            evaluationSeed: row.evaluationSeed,
            result: s.result_digest,
            request: s.request_hash,
            parameters: c.parameter_digest,
            tables: c.tables_digest,
          },
          sources,
        });
        return row;
      });
    const p = s.request.protocol;
    const costSource = source(
      "Recorded cost accounting",
      `${archive}/report.md`,
    );
    const costs: StudySummary["costs"] = [
      {
        label: "Finite training grid",
        value: s.costs.training.finite.endpoint_draws,
        unit: "endpoint draws",
        scope:
          "All six finite fits per seed, three seeds; occupancy, gradient and reference draws.",
        evidence: s.costs.evidence_class,
        source: costSource,
      },
      {
        label: "Equilibrium training",
        value: s.costs.training.equilibrium.endpoint_draws,
        unit: "endpoint draws",
        scope: "One oracle fit per seed, three seeds.",
        evidence: s.costs.evidence_class,
        source: costSource,
      },
      {
        label: "All training",
        value: s.costs.training.endpoint_draws,
        unit: "endpoint draws",
        scope: "21 fits and 105 updates.",
        evidence: s.costs.evidence_class,
        source: costSource,
      },
      {
        label: "Held-out evaluation",
        value: s.costs.evaluation.endpoint_draws,
        unit: "endpoint draws",
        scope: "All 60 cells.",
        evidence: s.costs.evidence_class,
        source: costSource,
      },
      {
        label: "Oracle modeled sweeps",
        value: s.costs.training.equilibrium.modeled_sweeps,
        unit: "sweeps",
        scope:
          "Unassigned: exact equilibrium sampling has no finite sweep cost.",
        evidence: s.costs.evidence_class,
        source: costSource,
      },
    ];
    const prov = z
      .object({
        generation_work: z.object({
          study_generation: z.object({ inclusive_seconds: num }),
        }),
      })
      .parse(parseEvidence(texts["provenance.json"]));
    costs.push({
      label: "Recorded CPU generation time",
      value: prov.generation_work.study_generation.inclusive_seconds,
      unit: "seconds",
      scope:
        "Inclusive study generation only; excludes separate preflight, persisted replay and final release verification. See provenance for runtime and timing policy. Not energy or device latency.",
      evidence: "software_simulation",
      source: source("Timing provenance", `${archive}/provenance.json`),
    });
    snapshot.study = {
      counts: {
        fits: s.fits.length,
        updates: s.fits.reduce((n, f) => n + f.steps.length, 0),
        cells: cells.length,
        pairs: s.evaluations.reduce((n, e) => n + e.pairs.length, 0),
      },
      cells,
      thresholds: {
        loss: p.loss_max,
        leakage: p.leakage_max,
        survival: p.survival_min,
        hopMae: p.hop_mae_max,
        asymmetryMae: p.asymmetry_mae_max,
      },
      costs,
      pairedDiagnostics: s.evaluations.flatMap((e) =>
        e.pairs.map((pair, i) => ({
          label: `Seed ${e.seed} · K${[1, 2, 4, 8, 16, 30][i]}`,
          estimate:
            pair.statistics.population_objective_difference_after_minus_before,
          interval: pair.statistics.paired_jackknife_normal_95_interval,
          conclusion: pair.statistics.population_objective_conclusion,
        })),
      ),
      sources,
    };
    snapshot.availability = "available";
    snapshot.execution = {
      state: "complete",
      label: "Complete",
      source: source("Final completion", `${archive}/completion.json`),
    };
    snapshot.verification = {
      state: "recorded",
      label: "Recorded",
      source: source("Recorded verification", `${archive}/verification.md`),
    };
    snapshot.science = {
      state: "quality_failure",
      label: "Joint quality unmet",
      source: studySource,
    };
  } catch (error) {
    snapshot.issues.push(
      error instanceof z.ZodError
        ? "Unsupported evidence structure"
        : error instanceof Error && error.message.includes("archive")
          ? error.message
          : "Evidence missing, unreadable or unsupported",
    );
  }
  snapshot.milestones = milestones.map((m) =>
    m.id === "M4G"
      ? {
          ...m,
          state:
            snapshot.execution.state === "complete"
              ? "Complete"
              : "Unavailable",
          conclusion:
            snapshot.execution.state === "complete"
              ? m.conclusion
              : "Evidence unavailable: completion and scientific status are withheld until the supported archive is authenticated.",
        }
      : m,
  );
  const recentMilestones = snapshot.recentStudies
    .slice()
    .reverse()
    .map((study) => ({
      id: study.id,
      question: study.title,
      conclusion: study.finding,
      state: study.availability === "available" ? "Complete" : "Unavailable",
      evidenceClasses: ["exact_reference"],
      sources: study.sources,
    }));
  snapshot.milestones.splice(
    snapshot.milestones.length - 1,
    0,
    ...recentMilestones,
    {
      id: "Next",
      question: "Direct local-fitting diagnostic",
      conclusion:
        "Proposed: fit the complete target table with a larger fixed budget and predetermined starts, then evaluate the same three-operation circuit. No new run has been executed.",
      state: "Proposed",
      evidenceClasses: [],
      sources: snapshot.recentStudies[0].sources.slice(0, 1),
    },
  );
  snapshot.research = [
    ...snapshot.recentStudies
      .filter((s) => s.availability === "available")
      .map((study) => ({
        id: study.id,
        kind: "finding" as const,
        title: study.title,
        text: study.finding,
        scope: study.scope,
        sources: study.sources,
      })),
    ...research,
  ];
  return { snapshot, details };
}
