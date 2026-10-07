# Associative memory stage A2: label-first categorical reference

Reference: THRML 0.1.4 categorical node, label updated first (`software_simulation`).
Binary arms: stage A held-out results, authenticated by SHA-256. No hardware claim.

Request digest `sha256:650a3829e916aae7fa2d0e25ed0318aef3ffe095d4cc01accff1f6abc3cac34e`; result digest `sha256:100fa335fd7219f47ef04698b290e791723585972b403a06c6e489d7a3e90acf`;
stage A archive `c83ee9dcd33c0a698ed03f60913f869ffa9088355902c7d1f07bde1fbd31b26f`.

## Reference recall (held-out) and the stage A confound

| cell | K | label-first | on target label | stage A reference | difference [95% CI] |
| --- | --- | --- | --- | --- | --- |
| P8/c12 | 4 | 0.912 | 0.854 | 0.779 | +0.132 [+0.123, +0.142] |
| P8/c12 | 16 | 0.899 | 0.891 | 0.863 | +0.036 [+0.027, +0.047] |
| P8/c12 | 64 | 0.918 | 0.923 | 0.904 | +0.014 [+0.007, +0.022] |
| P8/c12 | 256 | 0.936 | 0.954 | 0.931 | +0.005 [-0.000, +0.012] |
| P8/c8 | 4 | 0.782 | 0.601 | 0.672 | +0.110 [+0.102, +0.118] |
| P8/c8 | 16 | 0.780 | 0.645 | 0.768 | +0.012 [+0.003, +0.022] |
| P8/c8 | 64 | 0.800 | 0.690 | 0.807 | -0.007 [-0.020, +0.005] |
| P8/c8 | 256 | 0.834 | 0.761 | 0.791 | +0.043 [+0.026, +0.057] |
| P32/c12 | 4 | 0.821 | 0.724 | 0.751 | +0.070 [+0.062, +0.080] |
| P32/c12 | 16 | 0.857 | 0.797 | 0.829 | +0.028 [+0.020, +0.035] |
| P32/c12 | 64 | 0.896 | 0.873 | 0.884 | +0.012 [+0.007, +0.017] |
| P32/c12 | 256 | 0.931 | 0.938 | 0.923 | +0.008 [+0.004, +0.013] |
| P32/c8 | 4 | 0.658 | 0.347 | 0.614 | +0.044 [+0.036, +0.052] |
| P32/c8 | 16 | 0.710 | 0.640 | 0.704 | +0.007 [+0.001, +0.012] |
| P32/c8 | 64 | 0.747 | 0.728 | 0.748 | -0.002 [-0.007, +0.003] |
| P32/c8 | 256 | 0.752 | 0.584 | 0.749 | +0.003 [-0.014, +0.020] |
| P128/c12 | 4 | 0.708 | 0.504 | 0.671 | +0.037 [+0.028, +0.046] |
| P128/c12 | 16 | 0.769 | 0.797 | 0.762 | +0.006 [+0.001, +0.012] |
| P128/c12 | 64 | 0.817 | 0.726 | 0.813 | +0.004 [-0.003, +0.012] |
| P128/c12 | 256 | 0.888 | 0.860 | 0.884 | +0.004 [-0.002, +0.011] |
| P128/c8 | 4 | 0.559 | 0.200 | 0.549 | +0.010 [+0.004, +0.016] |
| P128/c8 | 16 | 0.596 | 0.300 | 0.594 | +0.002 [-0.003, +0.007] |
| P128/c8 | 64 | 0.608 | 0.331 | 0.610 | -0.002 [-0.007, +0.002] |
| P128/c8 | 256 | 0.641 | 0.348 | 0.637 | +0.004 [-0.005, +0.014] |

## Binary arms against the label-first reference

| cell | K | onehot | domainwall | bias | hopfield |
| --- | --- | --- | --- | --- | --- |
| P8/c12 | 4 | 0.908 (-0.004, matches) | 0.677 (-0.234, falls_short) | 0.917 (+0.006, matches) | 0.853 (-0.059, inconclusive) |
| P8/c12 | 16 | 0.927 (+0.028, inconclusive) | 0.712 (-0.188, falls_short) | 0.972 (+0.073, exceeds) | 0.867 (-0.032, inconclusive) |
| P8/c12 | 64 | 0.941 (+0.023, inconclusive) | 0.738 (-0.180, falls_short) | 0.982 (+0.064, exceeds) | 0.878 (-0.040, inconclusive) |
| P8/c12 | 256 | 0.949 (+0.013, inconclusive) | 0.757 (-0.179, falls_short) | 0.985 (+0.048, exceeds) | 0.885 (-0.052, inconclusive) |
| P8/c8 | 4 | 0.780 (-0.002, inconclusive) | 0.632 (-0.150, falls_short) | 0.796 (+0.014, inconclusive) | 0.762 (-0.020, inconclusive) |
| P8/c8 | 16 | 0.821 (+0.040, inconclusive) | 0.675 (-0.105, falls_short) | 0.859 (+0.079, exceeds) | 0.767 (-0.014, inconclusive) |
| P8/c8 | 64 | 0.859 (+0.059, exceeds) | 0.666 (-0.134, falls_short) | 0.930 (+0.130, exceeds) | 0.768 (-0.032, inconclusive) |
| P8/c8 | 256 | 0.840 (+0.005, inconclusive) | 0.681 (-0.153, falls_short) | 0.943 (+0.109, exceeds) | 0.767 (-0.067, inconclusive) |
| P32/c12 | 4 | 0.845 (+0.024, inconclusive) | 0.594 (-0.227, falls_short) | 0.814 (-0.007, inconclusive) | 0.666 (-0.154, falls_short) |
| P32/c12 | 16 | 0.873 (+0.017, inconclusive) | 0.603 (-0.254, falls_short) | 0.950 (+0.093, exceeds) | 0.667 (-0.190, falls_short) |
| P32/c12 | 64 | 0.884 (-0.012, inconclusive) | 0.615 (-0.280, falls_short) | 0.961 (+0.065, exceeds) | 0.668 (-0.228, falls_short) |
| P32/c12 | 256 | 0.890 (-0.041, inconclusive) | 0.618 (-0.313, falls_short) | 0.982 (+0.051, exceeds) | 0.667 (-0.264, falls_short) |
| P32/c8 | 4 | 0.729 (+0.071, exceeds) | 0.558 (-0.100, falls_short) | 0.687 (+0.029, inconclusive) | 0.591 (-0.067, falls_short) |
| P32/c8 | 16 | 0.764 (+0.054, exceeds) | 0.573 (-0.137, falls_short) | 0.738 (+0.027, inconclusive) | 0.592 (-0.119, falls_short) |
| P32/c8 | 64 | 0.781 (+0.034, inconclusive) | 0.583 (-0.163, falls_short) | 0.858 (+0.112, exceeds) | 0.604 (-0.142, falls_short) |
| P32/c8 | 256 | 0.799 (+0.047, inconclusive) | 0.594 (-0.158, falls_short) | 0.899 (+0.147, exceeds) | 0.607 (-0.145, falls_short) |
| P128/c12 | 4 | 0.785 (+0.078, exceeds) | 0.549 (-0.158, falls_short) | 0.722 (+0.014, inconclusive) | 0.605 (-0.103, falls_short) |
| P128/c12 | 16 | 0.826 (+0.058, exceeds) | 0.566 (-0.202, falls_short) | 0.842 (+0.073, exceeds) | 0.606 (-0.163, falls_short) |
| P128/c12 | 64 | 0.842 (+0.025, inconclusive) | 0.579 (-0.238, falls_short) | 0.892 (+0.075, exceeds) | 0.607 (-0.210, falls_short) |
| P128/c12 | 256 | 0.850 (-0.039, falls_short) | 0.588 (-0.300, falls_short) | 0.972 (+0.084, exceeds) | 0.609 (-0.279, falls_short) |
| P128/c8 | 4 | 0.683 (+0.124, exceeds) | 0.540 (-0.019, inconclusive) | 0.566 (+0.007, matches) | 0.568 (+0.009, inconclusive) |
| P128/c8 | 16 | 0.705 (+0.109, exceeds) | 0.550 (-0.046, falls_short) | 0.618 (+0.022, inconclusive) | 0.573 (-0.023, inconclusive) |
| P128/c8 | 64 | 0.716 (+0.109, exceeds) | 0.567 (-0.041, inconclusive) | 0.672 (+0.064, exceeds) | 0.575 (-0.033, inconclusive) |
| P128/c8 | 256 | 0.723 (+0.082, exceeds) | 0.579 (-0.062, falls_short) | 0.711 (+0.070, exceeds) | 0.576 (-0.065, falls_short) |
