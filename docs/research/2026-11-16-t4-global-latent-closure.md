# The `t4` global-latent line is closed for every construction

2026-11-16. **An owner decision, not evidence.** This note records the owner's
answer to the premise question in direction review D-0011 (PR #162). It adds
no result. Every number below is quoted from the cited sources with their own
evidence class.

## Decision

D-0011 asked: does P-0013 close the global-latent line on `t4_h05_q035` for
every construction, so that no fourth variant is proposed under any framing?
The listed framings were a frozen-pixel or sequential-phase schedule, a
64-pixel staleness number, a weaker-coupling instance, and relay placement for
k = 16. The alternative was to also price the schedule feature as a desk
note.

The owner chose the first option in a THERMES session on 2026-11-16:

- **The line is closed for every construction.** No fourth variant is to be
  proposed under any framing, including those listed above. A row that would
  carry a global latent on `t4_h05_q035`, or reopen the denoising line by
  another route, needs a new explicit owner selection through the queue. It
  is not a director or proposer suggestion.
- **"An update schedule other than two colours" stays recorded as a
  conditional hardware requirement and nothing more.** It is already feature
  7 in [Z1 features that would change an answer](2026-11-13-z1-feature-requirements.md).
  No desk note pricing it was requested, and no row follows from it.

This does not change D-0005's denoising closure, which already stood
(`PROJECT_CHARTER.md`, "Commitments and evaluation"). It removes the
remaining framings D-0011 listed. No queue row changes: `dtm-global-latent` is
already dropped and `directed-relay-latent` is done.

## What the line established, and what it did not

Three constructions missed the 0.01 posterior-error gate under the Z1's
two-colour schedule, each for a different named reason:

| Construction | Miss | Named reason | Source |
| --- | --- | --- | --- |
| Trained local-latent denoiser (P-0007) | 0.078 to 0.147 above Bayes | Model class | [DTM baseline headroom](../experiment-reports/2026-10-16-dtm-baseline-headroom/summary.md), `exact_reference` and `software_simulation` |
| Hand-built tied-copy latent (P-0008) | 0.0117 against 0.01 | Finite tie | [DTM exact-posterior lattice](../experiment-reports/2026-10-17-dtm-exact-posterior-lattice/summary.md), `exact_reference` and `software_simulation` |
| Directed relay tree, reduced 2 × 3 instance (P-0013) | 0.059 to 0.079; 0.065 or more with ideal relays | Staleness | [P-0013 probe note](2026-11-13-directed-relay-latent-probe.md), exploration, exact stationary law |

Not established, and now not to be pursued under this line:

- The staleness bias at the full 64-pixel instance, where the latent couples
  more weakly to each pixel. It was not computed.
- Any schedule other than two colours. With ideal relays, a sequential order
  is exact, but the Z1 has no sequential order. A frozen-pixel schedule
  would cost extra phases or clamp writes, each a full write under the sealed
  profile.
- Relay placement for k = 16, which the archived placement cannot reach.
- One-way couplings were assumed throughout. They are a relaxed constraint
  (rule 8), with no published cost, precision or placement rule.

The closure is a decision about where effort goes. It is not a proof that no
construction could ever carry the latent.
