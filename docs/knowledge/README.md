# Thermodynamic knowledge base

*Reference material for people and agents working on Thermo. Started
September 29, 2026. Nothing here is recorded evidence: a source card says what
a source claims and releases, and whether Thermo has checked it.*

## Why it exists

Thermo's rules separate exact references, software simulations, calibrated
projections and hardware measurements ([evidence policy](../evidence-policy.md)).
Outside papers and libraries need the same discipline one level up. A paper's
projected energy advantage, a library's exact simulator and a result Thermo
reproduced are three different things, and an agent reading a summary should
not have to rediscover which is which.

## Layout

| Path | Contents |
| --- | --- |
| [`sources/`](sources/) | One card per source (paper, library, blog post or task catalog): a checked TOML header with the source's version and claims, then prose on method, relevance to Thermo and cautions. |
| [`concepts.md`](concepts.md) | Shared vocabulary, each term tied to where Thermo uses it. |
| [`experiment-backlog.md`](experiment-backlog.md) | Open questions and ranked candidate experiments, with three draft protocol sketches (E0, E1, E2). |
| [`lessons.md`](lessons.md) | Dated lessons from Thermo's own recorded studies and reviews, each with its evidence class as the source states it and a link to the report or note. |

## Claim status

Every card records a status for the source as a whole, and one for each of its
claims. A status describes Thermo's relationship to the source, and is separate
from the evidence class of any Thermo result.

| Status | Meaning |
| --- | --- |
| `asserted` | Stated in the source. Thermo has not checked it. |
| `code_released` | Public code exists. Thermo has not run or pinned it. |
| `pinned_dependency` | Locked in `uv.lock` with upstream contract tests (AGENTS.md rule 6). |
| `reproduced` | A Thermo record or report checks the claim. Cite the report and its evidence class. |

A card may give different statuses to different claims, for example a
`pinned_dependency` library whose paper's speedup is only `asserted`.

## Source index

| Card | Kind | Status in Thermo |
| --- | --- | --- |
| [Thermalizing Stochastic Programs](sources/extropic-thermalizers.md) | Paper | `reproduced` for methods (M1–M5) under reading B (decided 2026-10-03), with recorded ambiguities |
| [THRML](sources/extropic-thrml.md) | Library | `pinned_dependency` (0.1.4) |
| [Torx](sources/extropic-torx.md) | Library | `pinned_dependency` (0.0.1; 0.0.2 released 2026-09-30) |
| [Probabilistic hardware for diffusion-like models](sources/extropic-dtm-hardware.md) | Paper | `asserted` |
| [Training thermodynamic computers by gradient descent](sources/whitelam-gradient-descent.md) | Paper + code | `asserted`; code has no license |
| [Generative thermodynamic computing](sources/whitelam-generative.md) | Paper | `asserted` |
| [Online local learning for generative thermodynamic computing](sources/wang-deng-online-learning.md) | Paper | `asserted` |
| [Thermodynamic linear algebra](sources/aifer-thermodynamic-linear-algebra.md) | Paper | `asserted` |
| [thermox](sources/normal-thermox.md) | Library | `code_released` |
| [posteriors](sources/normal-posteriors.md) | Library | `code_released`; PyTorch, reference only |
| [First sparks of thermodynamic recursive intelligence](sources/extropic-thermo-rsi.md) | Blog post | `asserted`; weights, tasks and scoring unreleased |
| [hinton-problems](sources/cybertronai-hinton-problems.md) | Task catalog | `code_released`; Unlicense |
| [A framework for stochastic differentiable programming](sources/extropic-torx-paper.md) | Paper | `asserted`; the Torx paper; its API names differ from the releases |
| [Z1T: sparse transformer-like models](sources/extropic-z1t.md) | Blog post | `asserted`; energy figures are projections |
| [Dense Associative Memory for Pattern Recognition](sources/krotov-hopfield-dense-memory.md) | Paper | `asserted`; capacity ~N^(n-1) for polynomial energies |
| [On a model of associative memory with huge storage capacity](sources/demircigil-huge-capacity.md) | Paper | `asserted`; exponential capacity for F(x) = e^x |
| [Large Associative Memory Problem in Neurobiology and Machine Learning](sources/krotov-hopfield-large-memory.md) | Paper | `asserted`; dense memory from two-body synapses plus hidden neurons |
| [On the equivalence of Hopfield Networks and Boltzmann Machines](sources/barra-hopfield-boltzmann.md) | Paper | `asserted`; Gaussian-hidden RBM equals Hebbian Hopfield |
| [Domain wall encoding of discrete variables for quantum annealing and QAOA](sources/chancellor-domain-wall.md) | Paper | `asserted`; pairwise categorical encoding in P − 1 spins |
| [On the role of non-linear latent features in bipartite generative neural networks](sources/bonnaire-latent-features.md) | Paper | `asserted`; binary hidden units cut RBM memory capacity |
| [Thermodynamic significance of QUBO encoding on quantum annealers](sources/doucet-qubo-encoding.md) | Paper | `asserted`; one-hot penalty strength trade-off |

## Adding or updating a card

Each card is `sources/<id>.md` and opens with its title, a blank line and a
fenced `toml` block:

```toml
id = "author-short-name"           # must equal the file name
title = "Exact title"              # must equal the heading
kind = "paper"                     # paper | library | blog_post | task_catalog
status = "asserted"                # the source as a whole; see the table above
last_checked = 2026-10-01

[source]
url = "https://arxiv.org/abs/2501.00001v2"   # arXiv links name the version read
version = "v2"                     # arXiv vN, release, commit, or date and hash
released = 2026-01-15              # date of that version
authors = "Surname and Surname"
license = "Apache-2.0"             # libraries and task catalogs; "none" if absent
runtime = "JAX (requires ...)"     # libraries and task catalogs
pypi = "package"                   # optional; enables the upstream check

[[claims]]
id = "C1"
status = "asserted"
text = "One claim, with its conditions and numbers."
reports = []                       # required when status = "reproduced"
```

Optional tables are `[code]` (`url`, `card`, `license`, `note`) for a paper's
code and `[dependency]` (`package`, `version`), which is required exactly when
the status is `pinned_dependency`.

1. Read the exact version you link, and set `last_checked`.
2. Write each numeric claim as its own `[[claims]]` entry with its conditions.
   Mark it `asserted` until a Thermo report checks it.
3. Record the license and runtime before anyone proposes a dependency. Code
   with no license can be read but not copied into Thermo.
4. When a Thermo report checks a claim, set it to `reproduced` and list the
   report under `reports`. Negative results count.
5. Add an index row whose status starts with the card's status, and, if the
   source raises a question, update the backlog.
6. Run the checker. CI runs it through `tests/unit/test_knowledge_base.py`.

```bash
uv run python -m thermo_lab.knowledge_base                     # structure; no network
uv run python -m thermo_lab.knowledge_base --stale-after 90    # also list old cards
uv run python -m thermo_lab.knowledge_base --upstream          # newer versions; network
```

The structural check rejects a card whose pin disagrees with `pyproject.toml`
or `uv.lock`, a cited report that doesn't exist, a broken link, or a card
missing from the index. The upstream check compares PyPI releases, arXiv
versions and repository commits with what each card records.

Promoting a backlog item into a study follows the usual path: a dated research
note, a frozen protocol under `docs/experiments/`, then a roadmap row. The
backlog is not a schedule.

The [October sampling synthesis](../research/2026-10-02-sampling-synthesis.md)
connects the newly recorded native-inference experiments to this public-source
context. It keeps reproduced CPU findings separate from source-card hardware
claims and updates which research questions remain open.
