# Associative memory stage A: binary emulation of a categorical hidden unit

Exact references: float64 enumeration, hidden layers summed exactly (`exact_reference`).
Sampled cells: THRML 0.1.4 on CPU, float32 (`software_simulation`). Sweeps are
algorithmic counts. No hardware claim.

Request digest `sha256:56c05e28b1191b2d667774a99e803a9796bb06848f9baf22561f6626b549c4d3`; result digest `sha256:29790929e6936bbd1ebf3fdddb913c9eadacb6346f8c65945a6fdbd5568d7f82`.

## Held-out recall after K sweeps (difference vs categorical, 95% bootstrap interval)

### P8/c12

| arm | K=4 | K=16 | K=64 | K=256 | equilibrium (exact) |
| --- | --- | --- | --- | --- | --- |
| categorical | 0.779 | 0.863 | 0.904 | 0.931 | 0.999 (categorical/b16) |
| onehot | 0.908 (+0.128 [+0.114, +0.142] exceeds) | 0.927 (+0.064 [+0.040, +0.088] exceeds) | 0.941 (+0.037 [+0.016, +0.064] inconclusive) | 0.949 (+0.018 [-0.006, +0.047] inconclusive) | 0.999 (onehot/b16/f1.6) |
| domainwall | 0.677 (-0.102 [-0.123, -0.077] falls_short) | 0.712 (-0.152 [-0.175, -0.126] falls_short) | 0.738 (-0.166 [-0.194, -0.132] falls_short) | 0.757 (-0.174 [-0.204, -0.140] falls_short) | 0.999 (domainwall/b16/f1.6) |
| bias | 0.917 (+0.138 [+0.122, +0.154] exceeds) | 0.972 (+0.109 [+0.093, +0.126] exceeds) | 0.982 (+0.079 [+0.064, +0.093] exceeds) | 0.985 (+0.054 [+0.044, +0.064] exceeds) | 0.999 (bias/b16/f0.6) |
| hopfield | 0.853 (+0.074 [+0.027, +0.111] exceeds) | 0.867 (+0.004 [-0.049, +0.046] inconclusive) | 0.878 (-0.026 [-0.085, +0.020] inconclusive) | 0.885 (-0.046 [-0.103, -0.001] inconclusive) | 0.896 (hopfield/b16) |

### P8/c8

| arm | K=4 | K=16 | K=64 | K=256 | equilibrium (exact) |
| --- | --- | --- | --- | --- | --- |
| categorical | 0.672 | 0.768 | 0.807 | 0.791 | 0.998 (categorical/b16) |
| onehot | 0.780 (+0.108 [+0.080, +0.139] exceeds) | 0.821 (+0.053 [+0.014, +0.085] inconclusive) | 0.859 (+0.052 [+0.017, +0.081] inconclusive) | 0.840 (+0.048 [+0.008, +0.082] inconclusive) | 0.998 (onehot/b16/f1.6) |
| domainwall | 0.632 (-0.040 [-0.066, -0.016] inconclusive) | 0.675 (-0.093 [-0.114, -0.072] falls_short) | 0.666 (-0.141 [-0.168, -0.115] falls_short) | 0.681 (-0.110 [-0.145, -0.074] falls_short) | 0.998 (domainwall/b16/f1.6) |
| bias | 0.796 (+0.125 [+0.113, +0.134] exceeds) | 0.859 (+0.091 [+0.076, +0.107] exceeds) | 0.930 (+0.122 [+0.112, +0.131] exceeds) | 0.943 (+0.152 [+0.137, +0.166] exceeds) | 0.998 (bias/b16/f0.6) |
| hopfield | 0.762 (+0.090 [+0.038, +0.142] exceeds) | 0.767 (-0.001 [-0.055, +0.050] inconclusive) | 0.768 (-0.039 [-0.093, +0.011] inconclusive) | 0.767 (-0.025 [-0.092, +0.037] inconclusive) | 0.754 (hopfield/b16) |

### P32/c12

| arm | K=4 | K=16 | K=64 | K=256 | equilibrium (exact) |
| --- | --- | --- | --- | --- | --- |
| categorical | 0.751 | 0.829 | 0.884 | 0.923 | 0.992 (categorical/b16) |
| onehot | 0.845 (+0.094 [+0.068, +0.122] exceeds) | 0.873 (+0.044 [+0.009, +0.078] inconclusive) | 0.884 (+0.000 [-0.031, +0.028] inconclusive) | 0.890 (-0.033 [-0.058, -0.007] inconclusive) | 0.992 (onehot/b16/f1.6) |
| domainwall | 0.594 (-0.157 [-0.184, -0.133] falls_short) | 0.603 (-0.226 [-0.255, -0.201] falls_short) | 0.615 (-0.269 [-0.309, -0.231] falls_short) | 0.618 (-0.304 [-0.351, -0.262] falls_short) | 0.992 (domainwall/b16/f0.4) |
| bias | 0.814 (+0.064 [+0.043, +0.080] exceeds) | 0.950 (+0.121 [+0.105, +0.134] exceeds) | 0.961 (+0.077 [+0.065, +0.089] exceeds) | 0.982 (+0.059 [+0.044, +0.076] exceeds) | 0.991 (bias/b16/f0.8) |
| hopfield | 0.666 (-0.084 [-0.138, -0.034] falls_short) | 0.667 (-0.162 [-0.215, -0.111] falls_short) | 0.668 (-0.216 [-0.273, -0.162] falls_short) | 0.667 (-0.256 [-0.317, -0.196] falls_short) | 0.648 (hopfield/b4) |

### P32/c8

| arm | K=4 | K=16 | K=64 | K=256 | equilibrium (exact) |
| --- | --- | --- | --- | --- | --- |
| categorical | 0.614 | 0.704 | 0.748 | 0.749 | 0.975 (categorical/b16) |
| onehot | 0.729 (+0.115 [+0.092, +0.139] exceeds) | 0.764 (+0.061 [+0.029, +0.094] exceeds) | 0.781 (+0.033 [-0.001, +0.066] inconclusive) | 0.799 (+0.051 [+0.017, +0.086] inconclusive) | 0.975 (onehot/b16/f1.6) |
| domainwall | 0.558 (-0.056 [-0.067, -0.046] falls_short) | 0.573 (-0.130 [-0.144, -0.117] falls_short) | 0.583 (-0.165 [-0.178, -0.153] falls_short) | 0.594 (-0.155 [-0.170, -0.141] falls_short) | 0.975 (domainwall/b16/f0.4) |
| bias | 0.687 (+0.073 [+0.066, +0.079] exceeds) | 0.738 (+0.034 [+0.026, +0.042] exceeds) | 0.858 (+0.110 [+0.098, +0.120] exceeds) | 0.899 (+0.150 [+0.130, +0.165] exceeds) | 0.966 (bias/b16/f0.6) |
| hopfield | 0.591 (-0.023 [-0.058, +0.015] inconclusive) | 0.592 (-0.112 [-0.148, -0.072] falls_short) | 0.604 (-0.144 [-0.190, -0.092] falls_short) | 0.607 (-0.142 [-0.187, -0.090] falls_short) | 0.583 (hopfield/b16) |

### P128/c12

| arm | K=4 | K=16 | K=64 | K=256 | equilibrium (exact) |
| --- | --- | --- | --- | --- | --- |
| categorical | 0.671 | 0.762 | 0.813 | 0.884 | 0.993 (categorical/b16) |
| onehot | 0.785 (+0.114 [+0.098, +0.128] exceeds) | 0.826 (+0.064 [+0.048, +0.079] exceeds) | 0.842 (+0.028 [+0.008, +0.048] inconclusive) | 0.850 (-0.034 [-0.051, -0.018] inconclusive) | 0.993 (onehot/b16/f1.6) |
| domainwall | 0.549 (-0.121 [-0.137, -0.105] falls_short) | 0.566 (-0.196 [-0.212, -0.182] falls_short) | 0.579 (-0.235 [-0.250, -0.218] falls_short) | 0.588 (-0.295 [-0.306, -0.284] falls_short) | 0.993 (domainwall/b16/f1.6) |
| bias | 0.722 (+0.051 [+0.042, +0.060] exceeds) | 0.842 (+0.079 [+0.070, +0.088] exceeds) | 0.892 (+0.079 [+0.064, +0.093] exceeds) | 0.972 (+0.088 [+0.075, +0.101] exceeds) | 0.992 (bias/b16/f0.8) |
| hopfield | 0.605 (-0.066 [-0.110, -0.024] falls_short) | 0.606 (-0.157 [-0.200, -0.114] falls_short) | 0.607 (-0.206 [-0.243, -0.168] falls_short) | 0.609 (-0.275 [-0.311, -0.238] falls_short) | 0.615 (hopfield/b4) |

### P128/c8

| arm | K=4 | K=16 | K=64 | K=256 | equilibrium (exact) |
| --- | --- | --- | --- | --- | --- |
| categorical | 0.549 | 0.594 | 0.610 | 0.637 | 0.877 (categorical/b16) |
| onehot | 0.683 (+0.134 [+0.118, +0.148] exceeds) | 0.705 (+0.112 [+0.096, +0.125] exceeds) | 0.716 (+0.106 [+0.086, +0.124] exceeds) | 0.723 (+0.085 [+0.063, +0.108] exceeds) | 0.877 (onehot/b16/f1.6) |
| domainwall | 0.540 (-0.009 [-0.019, +0.001] matches) | 0.550 (-0.043 [-0.056, -0.030] falls_short) | 0.567 (-0.043 [-0.063, -0.023] falls_short) | 0.579 (-0.058 [-0.083, -0.032] falls_short) | 0.877 (domainwall/b16/f1.6) |
| bias | 0.566 (+0.017 [+0.008, +0.025] inconclusive) | 0.618 (+0.024 [+0.019, +0.029] inconclusive) | 0.672 (+0.062 [+0.053, +0.073] exceeds) | 0.711 (+0.074 [+0.052, +0.096] exceeds) | 0.874 (bias/b16/f0.8) |
| hopfield | 0.568 (+0.019 [-0.013, +0.053] inconclusive) | 0.573 (-0.021 [-0.054, +0.017] inconclusive) | 0.575 (-0.035 [-0.066, -0.002] inconclusive) | 0.576 (-0.061 [-0.090, -0.031] falls_short) | 0.596 (hopfield/b4) |

## Cost of the configuration selected at the largest budget

| cell | arm | config | hidden units | couplings | dynamic range | blocks per sweep | updates per sweep |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P8/c12 | categorical | categorical/b8 | 1 | 192 | 1.0 | 2 | 13 |
| P8/c12 | onehot | onehot/b16/f0.4 | 8 | 220 | 57.6 | 9 | 20 |
| P8/c12 | domainwall | domainwall/b8/f0.1 | 7 | 189 | 2.4 | 3 | 19 |
| P8/c12 | bias | bias/b16/f0.6 | 8 | 220 | 14.4 | 2 | 20 |
| P8/c12 | hopfield | hopfield/b8 | 0 | 276 | 3.9 | 12 | 12 |
| P8/c8 | categorical | categorical/b8 | 1 | 192 | 1.0 | 2 | 17 |
| P8/c8 | onehot | onehot/b16/f0.4 | 8 | 220 | 57.6 | 9 | 24 |
| P8/c8 | domainwall | domainwall/b8/f0.1 | 7 | 189 | 2.4 | 3 | 23 |
| P8/c8 | bias | bias/b16/f0.6 | 8 | 220 | 14.4 | 2 | 24 |
| P8/c8 | hopfield | hopfield/b8 | 0 | 276 | 3.9 | 16 | 16 |
| P32/c12 | categorical | categorical/b8 | 1 | 768 | 1.0 | 2 | 13 |
| P32/c12 | onehot | onehot/b16/f0.4 | 32 | 1264 | 288.0 | 33 | 44 |
| P32/c12 | domainwall | domainwall/b8/f0.1 | 31 | 1209 | 2.4 | 3 | 43 |
| P32/c12 | bias | bias/b16/f0.8 | 32 | 1264 | 19.2 | 2 | 44 |
| P32/c12 | hopfield | hopfield/b16 | 0 | 276 | 8.4 | 12 | 12 |
| P32/c8 | categorical | categorical/b4 | 1 | 768 | 1.0 | 2 | 17 |
| P32/c8 | onehot | onehot/b16/f0.4 | 32 | 1264 | 288.0 | 33 | 48 |
| P32/c8 | domainwall | domainwall/b4/f0.4 | 31 | 1209 | 9.6 | 3 | 47 |
| P32/c8 | bias | bias/b8/f0.6 | 32 | 1264 | 14.7 | 2 | 48 |
| P32/c8 | hopfield | hopfield/b4 | 0 | 276 | 8.4 | 16 | 16 |
| P128/c12 | categorical | categorical/b8 | 1 | 3072 | 1.0 | 2 | 13 |
| P128/c12 | onehot | onehot/b16/f0.4 | 128 | 11200 | 1209.6 | 129 | 140 |
| P128/c12 | domainwall | domainwall/b4/f0.4 | 127 | 11049 | 9.6 | 3 | 139 |
| P128/c12 | bias | bias/b16/f0.8 | 128 | 11200 | 25.6 | 2 | 140 |
| P128/c12 | hopfield | hopfield/b16 | 0 | 276 | 17.3 | 12 | 12 |
| P128/c8 | categorical | categorical/b8 | 1 | 3072 | 1.0 | 2 | 17 |
| P128/c8 | onehot | onehot/b16/f0.4 | 128 | 11200 | 1209.6 | 129 | 144 |
| P128/c8 | domainwall | domainwall/b4/f0.1 | 127 | 11049 | 2.4 | 3 | 143 |
| P128/c8 | bias | bias/b8/f0.6 | 128 | 11200 | 25.3 | 2 | 144 |
| P128/c8 | hopfield | hopfield/b4 | 0 | 276 | 17.3 | 16 | 16 |

## Range-sensitivity curve (held-out exact recall, best development config with D <= R)

### P8/c12

| arm | R=2 | R=4 | R=8 | R=16 | R=32 | R=64 | R=128 | R=inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| categorical | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| onehot | - | - | - | 0.771 | 0.771 | 0.961 | 0.961 | 0.999 |
| domainwall | - | 0.763 | 0.763 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| bias | - | - | 0.739 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| hopfield | - | 0.896 | 0.896 | 0.896 | 0.896 | 0.896 | 0.896 | 0.896 |

### P8/c8

| arm | R=2 | R=4 | R=8 | R=16 | R=32 | R=64 | R=128 | R=inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| categorical | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 |
| onehot | - | - | - | 0.781 | 0.781 | 0.962 | 0.962 | 0.998 |
| domainwall | - | 0.697 | 0.697 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 |
| bias | - | - | 0.752 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 |
| hopfield | - | 0.754 | 0.754 | 0.754 | 0.754 | 0.754 | 0.754 | 0.754 |

### P32/c12

| arm | R=2 | R=4 | R=8 | R=16 | R=32 | R=64 | R=128 | R=inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| categorical | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 |
| onehot | - | - | - | - | - | - | 0.761 | 0.992 |
| domainwall | - | 0.625 | 0.625 | 0.992 | 0.992 | 0.992 | 0.992 | 0.992 |
| bias | - | - | - | 0.991 | 0.991 | 0.991 | 0.991 | 0.991 |
| hopfield | - | - | - | 0.648 | 0.648 | 0.648 | 0.648 | 0.648 |

### P32/c8

| arm | R=2 | R=4 | R=8 | R=16 | R=32 | R=64 | R=128 | R=inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| categorical | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 |
| onehot | - | - | - | - | - | - | 0.708 | 0.975 |
| domainwall | - | 0.605 | 0.605 | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 |
| bias | - | - | - | 0.966 | 0.966 | 0.966 | 0.966 | 0.966 |
| hopfield | - | - | - | 0.583 | 0.583 | 0.583 | 0.583 | 0.583 |

### P128/c12

| arm | R=2 | R=4 | R=8 | R=16 | R=32 | R=64 | R=128 | R=inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| categorical | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 |
| onehot | - | - | - | - | - | - | - | 0.993 |
| domainwall | - | 0.609 | 0.609 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 |
| bias | - | - | - | - | 0.992 | 0.992 | 0.992 | 0.992 |
| hopfield | - | - | - | - | 0.615 | 0.615 | 0.615 | 0.615 |

### P128/c8

| arm | R=2 | R=4 | R=8 | R=16 | R=32 | R=64 | R=128 | R=inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| categorical | 0.877 | 0.877 | 0.877 | 0.877 | 0.877 | 0.877 | 0.877 | 0.877 |
| onehot | - | - | - | - | - | - | - | 0.877 |
| domainwall | - | 0.596 | 0.596 | 0.877 | 0.877 | 0.877 | 0.877 | 0.877 |
| bias | - | - | - | - | 0.874 | 0.874 | 0.874 | 0.874 |
| hopfield | - | - | - | - | 0.596 | 0.596 | 0.596 | 0.596 |

