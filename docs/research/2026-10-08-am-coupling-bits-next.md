# Associative-memory coupling bits: what changes next

*2026-10-08, THERMES. Restates the
[coupling-bits findings](../experiment-reports/2026-10-08-am-coupling-bits/findings.md);
not new evidence.*

- Program the bias associative-memory design with the codebook step equal to
  the coupling J. Fields become integer multiples of J. Six bits then keep
  exact equilibrium recall within 0.01 of unquantized in all six cells
  (`exact_reference`).
- Do not use one full-scale cap for this design. It needs 12 bits, because the
  largest field (up to 28 J at P = 128) sets the step. Separate coupling and
  field caps need 10; that number comes from a wide interval at P = 128 with an
  8-bit cue rather than a measured mean loss.
- Static ±10% per-site beta jitter is not a constraint for this design
  (`software_simulation`, all 240 intervals within ±0.01). Per-sweep beta noise
  and coupling-value noise remain untested.
- The codebook is a mathematical rounding rule, not a Z1 encoding. Whether Z1's
  programming path can place its step at J is a hardware question this study
  cannot answer.
