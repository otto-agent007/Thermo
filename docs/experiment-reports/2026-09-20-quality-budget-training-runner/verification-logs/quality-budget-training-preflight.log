# M4G training-law component preflight

Seven training laws pass the bounded three-site checks. **Full M4G runner readiness remains false.**

Request: `sha256:0a8496a7ace6115e52a1688002335ae906c2cf2eb9285f1d468402b23fe843ae`

Result: `sha256:07adfe6964346e9962301288a417381b1b13f962a5ea6a222c9e95cd322196dd`

| Law | Exact gradient error | Finite-difference error | Draw-replay error |
| --- | ---: | ---: | ---: |
| 1 | 6.66e-16 | 1.65e-10 | 0 |
| 2 | 8.33e-16 | 2.87e-10 | 0 |
| 4 | 1.83e-15 | 5.91e-10 | 0 |
| 8 | 3.39e-15 | 1.1e-09 | 0 |
| 16 | 7.44e-15 | 2.01e-09 | 0 |
| 30 | 1.44e-14 | 3.73e-09 | 0 |
| equilibrium | 1.67e-16 | 2.18e-10 | 0 |

All occupancy counts replay exactly. Gradient replay compares sums and sum-squares at atol 1e-10, rtol 1e-12; exact-gradient tolerance is 1e-12 and finite differences use 1e-7. These are numerical integrity tolerances, not relaxed quality thresholds.

Three independent diagnostic role pairs per law use 42 distinct seeds, disjoint from all 213 study roles. Mixed occupancy laws, equilibrium scores substituted at K1, and missing shared occurrences are rejected by exact negative controls.

Exact references are `exact_reference`; sampled role evidence is `software_simulation`. Per build, production sampling and independent replay each use 4,128,768 local endpoint draws. Report validation rebuilds the gate. Reference draws are local, not additional propagated trajectories.

Zero fits, parameter updates or held-out study cells executed. This validates sampling components on the fixed fixture, not the full training runner, task quality, inference savings, convergence or hardware performance. Next: integrate five-update training and complete its integrity preflight.
