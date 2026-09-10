# Frozen-pair finite-sweep transfer audit (M2)

Recorded on September 10, 2026 from the successful
[checked experiment job](https://github.com/otto-agent007/Thermo/actions/runs/34457547108/job/102807362606)
at implementation commit `76e17fded25b7226f1afa0e6ea5095a7eb81d1d2`.
The generated report below is retained without changing its numerical values.
The workflow's separate documentation assertion and formatting findings were
addressed afterward; this provenance identifies the completed experiment job,
not an all-green workflow claim.

[Download the M1 sources and full M2 artifacts](https://github.com/otto-agent007/Thermo/actions/runs/34457547108/artifacts/10144321952).
The archive SHA-256 is
`c7186334f9e857fd687cac9ef26f1a747a3467058cc65dbf1bca9a0e38022353`.
GitHub artifact retention for this run is 90 days; the metrics and scientific
identities below are committed permanently in this report.

All full-program results are software_simulation. Exact local joint endpoint tables and the target reference are exact_reference. Explicitly supplied frozen M1 pairs are evaluated without additional training. The equilibrium control reproduces M1 exactly.

Each cell uses 32,768 complete trajectory pairs over 25 sites and 500 occurrences. One PCG64 uniform vector per occurrence is shared across all 14 member/horizon cells. Each finite kernel resets uniformly and represents complete hidden-then-output sweeps. NumPy samples exact finite-horizon joint endpoints; no live Gibbs chain is executed.

The order-two U-statistic estimates squared population occupancy loss without finite-batch bias. Negative after-minus-before differences favor the updated pair. Approximate normal 95% intervals use paired delete-one jackknife uncertainty conditional on each frozen pair. Near-zero and tiny-sample coverage can be far below 95% (M1 toy examples: 50% and 62.5%). These are pointwise intervals, not simultaneous bands. All improved/regressed/inconclusive conclusions are descriptive and non-gating.

Horizons and paired members are correlated, not independent replications. Independent source seeds are the cross-run replication units. The reused M1 held-out stream is an equilibrium reproduction control, not fresh independent confirmation.

| Seed | Horizon | Before loss | After loss | After minus before | Approximate 95% interval | Conclusion | Leakage before | Leakage after | Leakage delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | equilibrium | 0.051450143146241044 | 0.05103811173615447 | -0.000412031410086576 | (-0.0007250239646529582, -9.903885552019377e-05) | improved | 0.59149169921875 | 0.590789794921875 | -0.000701904296875 |
| 0 | k1 | 0.23460062227312226 | 0.2343791653316066 | -0.00022145694151565198 | (-0.00040922771981441626, -3.368616321688768e-05) | improved | 0.846099853515625 | 0.84564208984375 | -0.000457763671875 |
| 0 | k2 | 0.2221796129139954 | 0.22214887558486446 | -3.073732913094762e-05 | (-0.0003394653808019009, 0.0002779907225400057) | inconclusive | 0.83154296875 | 0.83099365234375 | -0.00054931640625 |
| 0 | k4 | 0.18550395801876518 | 0.18540295882494595 | -0.00010099919381922895 | (-0.0005246962359653415, 0.0003226978483268836) | inconclusive | 0.763946533203125 | 0.7637939453125 | -0.000152587890625 |
| 0 | k8 | 0.12318764113958515 | 0.12260228589259074 | -0.0005853552469944068 | (-0.0010181427417812244, -0.00015256775220758912) | improved | 0.680877685546875 | 0.679901123046875 | -0.0009765625 |
| 0 | k16 | 0.06997403868895223 | 0.06946132379061465 | -0.0005127148983375823 | (-0.0008734907886984034, -0.0001519390079767612) | improved | 0.61651611328125 | 0.61676025390625 | 0.000244140625 |
| 0 | k30 | 0.05381073691962574 | 0.0532809401304478 | -0.0005297967891779443 | (-0.000862920707634347, -0.00019667287072154153) | improved | 0.596832275390625 | 0.5963134765625 | -0.000518798828125 |
| 1 | equilibrium | 0.04836867496940912 | 0.04784587483193872 | -0.0005228001374704 | (-0.0008193180461950576, -0.0002262822287457423) | improved | 0.590362548828125 | 0.590576171875 | 0.000213623046875 |
| 1 | k1 | 0.23379200533513506 | 0.23375589079430883 | -3.61145408262209e-05 | (-0.0002184474523953718, 0.00014621837074293) | inconclusive | 0.845306396484375 | 0.84490966796875 | -0.000396728515625 |
| 1 | k2 | 0.2215310418814746 | 0.2214075695548506 | -0.0001234723266239912 | (-0.0004558180633772908, 0.00020887341012930838) | inconclusive | 0.82720947265625 | 0.827362060546875 | 0.000152587890625 |
| 1 | k4 | 0.1832388451832741 | 0.18321555824641736 | -2.3286936856742324e-05 | (-0.0004500838643942641, 0.00040350999068077947) | inconclusive | 0.763519287109375 | 0.76422119140625 | 0.000701904296875 |
| 1 | k8 | 0.12197828679146691 | 0.12122150765970194 | -0.000756779131764973 | (-0.0012002270799422814, -0.0003133311835876646) | improved | 0.67852783203125 | 0.678955078125 | 0.00042724609375 |
| 1 | k16 | 0.06611746698271058 | 0.06552031129674499 | -0.0005971556859655919 | (-0.0009549765336683685, -0.0002393348382628153) | improved | 0.614105224609375 | 0.6146240234375 | 0.000518798828125 |
| 1 | k30 | 0.05022910812102977 | 0.04983825270027831 | -0.0003908554207514639 | (-0.0007042348775549464, -7.74759639479813e-05) | improved | 0.59527587890625 | 0.595001220703125 | -0.000274658203125 |
| 2 | equilibrium | 0.050518269369543 | 0.04995764645472812 | -0.0005606229148148803 | (-0.0008648132557958724, -0.00025643257383388814) | improved | 0.592803955078125 | 0.592620849609375 | -0.00018310546875 |
| 2 | k1 | 0.23556959944072073 | 0.23512554064748203 | -0.00044405879323869413 | (-0.0006808693142232633, -0.00020724827225412502) | improved | 0.846527099609375 | 0.8460693359375 | -0.000457763671875 |
| 2 | k2 | 0.2200653051354036 | 0.21977118493436132 | -0.000294120201042275 | (-0.0006279749407477911, 3.973453866324111e-05) | inconclusive | 0.825164794921875 | 0.824951171875 | -0.000213623046875 |
| 2 | k4 | 0.1826595248510203 | 0.18252888265238235 | -0.00013064219863795357 | (-0.0005149018620900582, 0.00025361746481415107) | inconclusive | 0.76177978515625 | 0.76165771484375 | -0.0001220703125 |
| 2 | k8 | 0.12157056207089087 | 0.12112677725094388 | -0.00044378481994698793 | (-0.0008266887931853707, -6.0880846708605166e-05) | improved | 0.678863525390625 | 0.677886962890625 | -0.0009765625 |
| 2 | k16 | 0.06843223826579645 | 0.06772666651760914 | -0.0007055717481873058 | (-0.001078108812063887, -0.0003330346843107248) | improved | 0.61474609375 | 0.613250732421875 | -0.001495361328125 |
| 2 | k30 | 0.05280726122876196 | 0.05236801683886723 | -0.0004392443898947329 | (-0.0007428435879088429, -0.00013564519188062293) | improved | 0.59515380859375 | 0.59442138671875 | -0.000732421875 |

## Across-seed descriptive means

Each mean weights the supplied independent source seeds equally. No per-run SEs or intervals are averaged, and no across-seed confidence claim is made.

| Horizon | Independent seeds | Mean loss difference (after minus before) |
| --- | --- | --- |
| equilibrium | 3 | -0.0004984848207906187 |
| k1 | 3 | -0.00023387675852685566 |
| k2 | 3 | -0.00014944328559907127 |
| k4 | 3 | -8.497610977130828e-05 |
| k8 | 3 | -0.0005953063995687893 |
| k16 | 3 | -0.0006051474441634933 |
| k30 | 3 | -0.000453298866608047 |

## Declared sampling work

Counts below are per complete trajectory per member of the frozen pair. Multiply by 2 x 32,768 for both members at one finite horizon and source seed. Three free pbits update per complete sweep. Equilibrium has no finite sweep budget.

| Horizon | Complete sweeps | Free pbit updates |
| --- | --- | --- |
| equilibrium | None | None |
| k1 | 500 | 1500 |
| k2 | 1000 | 3000 |
| k4 | 2000 | 6000 |
| k8 | 4000 | 12000 |
| k16 | 8000 | 24000 |
| k30 | 15000 | 45000 |

These are declared algorithmic counts, not measured device operations, energy, or latency. They exclude reset, clamp, parameter writes, readout, host I/O, and table setup. NumPy executes endpoint draws rather than these Gibbs updates.

## Identities and execution timing

Timing boundary: execution_seconds measures synchronous NumPy float64 local joint-table construction, PCG64 endpoint sampling over all 14 frozen-pair/horizon cells, and bounded terminal reductions; compile_seconds is zero because table construction is included in execution; excludes M1 source validation and lineage reconstruction, provenance, artifact validation, persistence, aggregation, and reporting; no live Gibbs, THRML, hosted or hardware execution

| Seed | Historical PR #20 pair | M1 source summary | M2 request | M2 result | NumPy seconds |
| --- | --- | --- | --- | --- | --- |
| 0 | False | sha256:c8cc525d4eff5f29110679d218cacef6344e1f00d3e32470ec9723116fc814a1 | sha256:31b269f5b8625f48fc0aea3b8052fd630ecec2f201f6726542b06f79459f58c5 | sha256:1638a707bf9763f0bc0a2d02511e85aaa2dae30ea26fbf8b49e843fab4889c7d | 38.740250385999985 |
| 1 | False | sha256:cebd290909149b08b21128c02cc28fcdb91d9931899eb7e206be09e9156580b7 | sha256:60a25bcb1b761779e81874746ec48059ce890ce4545367f5012fcce423d4be02 | sha256:e749a1a75c1ab2a3d5f45eda586d3284431339e731460c87950be915a24e1958 | 26.429434268000023 |
| 2 | False | sha256:d9978cde1fb7b63b5b883e153835af83cbf09c889d9e7c73bc434d359b329335 | sha256:62dfaeec7e70ad784d31d29ccec59ca768655702300f22f65d7d588915ecc256 | sha256:bf7a8ff4e87e2abcfecd19c5d802ee01dc5ba9d8a5b3455398c383d9fd8bb1b5 | 26.468343790000006 |

A regenerated numerical compiler lineage need not match the historical release bit for bit. The table explicitly marks historical identity matches. M2 freezes the supplied validated source parameters and reproduces that source's equilibrium control exactly; it does not silently substitute a historical pair.

Moment feasibility and digest consistency are necessary audit checks, not complete binary realizability or proof of execution. This study does not establish finite-sweep gradient correctness, iterative convergence, official Thermalizers compatibility, hosted simulation, or physical Z1/TSU performance.
