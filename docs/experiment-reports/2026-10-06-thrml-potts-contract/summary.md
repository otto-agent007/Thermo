# THRML categorical finite-sweep contract (Potts stage A / A3)

Exact references: float64 enumeration of the 729-state Potts patch and the
32-state E0 chain, and powers of their ordered block-Gibbs sweep matrices
(`exact_reference`). THRML cells: 0.1.4 on CPU, float32, 400000 independent chains
per cell (`software_simulation`). No hardware evidence anywhere in this report.

Request digest `sha256:293b307a547c00e17d75907da7fbfb78c8103d567bce69cf0d1352f480ed0139`; result digest `sha256:d06339e40e67805c9f7f59daae80d5889c92fd3c464d3d6193ef6fe0cc2b7aec`.

## Cells

| cell | TV to p0 T^K | tolerance (q=0.999) | pass | max marginal error | TV to p0 T^(K-1) | TV to p0 T^(K+1) | closest | TV to stationary (exact) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bridge/generic/forward/all_zero/K0 | 0.0000 | 0.0000 | yes | 0.0000 | n/a | 0.9343 | K (decisive) | 0.9603 |
| bridge/generic/forward/all_zero/K1 | 0.0032 | 0.0050 | yes | 0.0016 | 0.9344 | 0.1289 | K (decisive) | 0.1344 |
| bridge/generic/forward/all_zero/K16 | 0.0032 | 0.0050 | yes | 0.0014 | 0.0032 | 0.0032 | K+1 (undecided) | 0.0000 |
| bridge/generic/forward/all_zero/K2 | 0.0033 | 0.0049 | yes | 0.0008 | 0.1286 | 0.0073 | K (decisive) | 0.0067 |
| bridge/generic/forward/all_zero/K3 | 0.0022 | 0.0052 | yes | 0.0009 | 0.0057 | 0.0024 | K (undecided) | 0.0005 |
| bridge/generic/forward/all_zero/K4 | 0.0043 | 0.0049 | yes | 0.0005 | 0.0041 | 0.0043 | K-1 (undecided) | 0.0000 |
| bridge/generic/forward/all_zero/K8 | 0.0035 | 0.0049 | yes | 0.0018 | 0.0035 | 0.0035 | K+1 (undecided) | 0.0000 |
| bridge/square/forward/all_zero/K0 | 0.0000 | 0.0000 | yes | 0.0000 | n/a | 0.9343 | K (decisive) | 0.9603 |
| bridge/square/forward/all_zero/K1 | 0.0028 | 0.0050 | yes | 0.0005 | 0.9341 | 0.1284 | K (decisive) | 0.1344 |
| bridge/square/forward/all_zero/K16 | 0.0034 | 0.0050 | yes | 0.0015 | 0.0034 | 0.0034 | K-1 (undecided) | 0.0000 |
| bridge/square/forward/all_zero/K2 | 0.0036 | 0.0049 | yes | 0.0007 | 0.1283 | 0.0069 | K (decisive) | 0.0067 |
| bridge/square/forward/all_zero/K3 | 0.0040 | 0.0052 | yes | 0.0014 | 0.0084 | 0.0040 | K (undecided) | 0.0005 |
| bridge/square/forward/all_zero/K4 | 0.0032 | 0.0049 | yes | 0.0014 | 0.0030 | 0.0032 | K-1 (undecided) | 0.0000 |
| bridge/square/forward/all_zero/K8 | 0.0038 | 0.0049 | yes | 0.0013 | 0.0038 | 0.0038 | K+1 (undecided) | 0.0000 |
| clamped/generic/forward/all_zero/K0 | 0.0000 | 0.0000 | yes | 0.0000 | n/a | 0.9191 | K (decisive) | 0.9297 |
| clamped/generic/forward/all_zero/K1 | 0.0064 | 0.0080 | yes | 0.0019 | 0.9187 | 0.1677 | K (decisive) | 0.2439 |
| clamped/generic/forward/all_zero/K16 | 0.0075 | 0.0088 | yes | 0.0013 | 0.0075 | 0.0075 | K-1 (undecided) | 0.0000 |
| clamped/generic/forward/all_zero/K2 | 0.0075 | 0.0086 | yes | 0.0026 | 0.1684 | 0.0543 | K (decisive) | 0.0791 |
| clamped/generic/forward/all_zero/K3 | 0.0068 | 0.0085 | yes | 0.0011 | 0.0541 | 0.0194 | K (decisive) | 0.0253 |
| clamped/generic/forward/all_zero/K4 | 0.0074 | 0.0088 | yes | 0.0011 | 0.0187 | 0.0099 | K (undecided) | 0.0081 |
| clamped/generic/forward/all_zero/K8 | 0.0075 | 0.0089 | yes | 0.0010 | 0.0076 | 0.0075 | K+1 (undecided) | 0.0001 |
| clamped/square/forward/all_zero/K0 | 0.0000 | 0.0000 | yes | 0.0000 | n/a | 0.9191 | K (decisive) | 0.9297 |
| clamped/square/forward/all_zero/K1 | 0.0062 | 0.0080 | yes | 0.0021 | 0.9190 | 0.1680 | K (decisive) | 0.2439 |
| clamped/square/forward/all_zero/K16 | 0.0070 | 0.0088 | yes | 0.0010 | 0.0070 | 0.0070 | K+1 (undecided) | 0.0000 |
| clamped/square/forward/all_zero/K2 | 0.0063 | 0.0086 | yes | 0.0009 | 0.1683 | 0.0547 | K (decisive) | 0.0791 |
| clamped/square/forward/all_zero/K3 | 0.0072 | 0.0085 | yes | 0.0016 | 0.0555 | 0.0178 | K (decisive) | 0.0253 |
| clamped/square/forward/all_zero/K4 | 0.0070 | 0.0088 | yes | 0.0013 | 0.0181 | 0.0103 | K (undecided) | 0.0081 |
| clamped/square/forward/all_zero/K8 | 0.0079 | 0.0089 | yes | 0.0011 | 0.0079 | 0.0079 | K+1 (undecided) | 0.0001 |
| potts/generic/forward/all_zero/K0 | 0.0000 | 0.0000 | yes | 0.0000 | n/a | 0.9699 | K (decisive) | 0.9781 |
| potts/generic/forward/all_zero/K1 | 0.0097 | 0.0109 | yes | 0.0024 | 0.9695 | 0.2302 | K (decisive) | 0.2818 |
| potts/generic/forward/all_zero/K16 | 0.0103 | 0.0125 | yes | 0.0011 | 0.0103 | 0.0103 | K+1 (undecided) | 0.0000 |
| potts/generic/forward/all_zero/K2 | 0.0110 | 0.0122 | yes | 0.0010 | 0.2294 | 0.0456 | K (decisive) | 0.0602 |
| potts/generic/forward/all_zero/K3 | 0.0112 | 0.0125 | yes | 0.0019 | 0.0468 | 0.0168 | K (undecided) | 0.0187 |
| potts/generic/forward/all_zero/K4 | 0.0107 | 0.0124 | yes | 0.0016 | 0.0173 | 0.0117 | K (undecided) | 0.0069 |
| potts/generic/forward/all_zero/K8 | 0.0108 | 0.0123 | yes | 0.0017 | 0.0108 | 0.0109 | K-1 (undecided) | 0.0005 |
| potts/generic/forward/uniform/K0 | 0.0167 | 0.0185 | yes | 0.0016 | n/a | 0.6194 | K (decisive) | 0.6617 |
| potts/generic/forward/uniform/K1 | 0.0126 | 0.0130 | yes | 0.0026 | 0.6198 | 0.1121 | K (decisive) | 0.1703 |
| potts/generic/forward/uniform/K16 | 0.0106 | 0.0122 | yes | 0.0013 | 0.0106 | 0.0106 | K+1 (undecided) | 0.0000 |
| potts/generic/forward/uniform/K2 | 0.0116 | 0.0127 | yes | 0.0010 | 0.1123 | 0.0404 | K (decisive) | 0.0701 |
| potts/generic/forward/uniform/K3 | 0.0106 | 0.0124 | yes | 0.0014 | 0.0396 | 0.0215 | K (decisive) | 0.0352 |
| potts/generic/forward/uniform/K4 | 0.0114 | 0.0123 | yes | 0.0010 | 0.0206 | 0.0156 | K (undecided) | 0.0192 |
| potts/generic/forward/uniform/K8 | 0.0112 | 0.0125 | yes | 0.0007 | 0.0112 | 0.0114 | K-1 (undecided) | 0.0018 |
| potts/generic/reversed/all_zero/K0 | 0.0000 | 0.0000 | yes | 0.0000 | n/a | 0.9699 | K (decisive) | 0.9781 |
| potts/generic/reversed/all_zero/K1 | 0.0093 | 0.0108 | yes | 0.0011 | 0.9703 | 0.2778 | K (decisive) | 0.3472 |
| potts/generic/reversed/all_zero/K16 | 0.0100 | 0.0125 | yes | 0.0009 | 0.0100 | 0.0100 | K+1 (undecided) | 0.0000 |
| potts/generic/reversed/all_zero/K2 | 0.0110 | 0.0123 | yes | 0.0014 | 0.2778 | 0.0608 | K (decisive) | 0.1056 |
| potts/generic/reversed/all_zero/K3 | 0.0113 | 0.0125 | yes | 0.0015 | 0.0590 | 0.0283 | K (decisive) | 0.0551 |
| potts/generic/reversed/all_zero/K4 | 0.0107 | 0.0124 | yes | 0.0021 | 0.0298 | 0.0177 | K (decisive) | 0.0302 |
| potts/generic/reversed/all_zero/K8 | 0.0112 | 0.0124 | yes | 0.0008 | 0.0114 | 0.0113 | K (undecided) | 0.0029 |
| potts/generic/reversed/uniform/K0 | 0.0172 | 0.0184 | yes | 0.0015 | n/a | 0.6168 | K (decisive) | 0.6617 |
| potts/generic/reversed/uniform/K1 | 0.0110 | 0.0131 | yes | 0.0020 | 0.6155 | 0.1286 | K (decisive) | 0.1524 |
| potts/generic/reversed/uniform/K16 | 0.0104 | 0.0124 | yes | 0.0009 | 0.0104 | 0.0104 | K-1 (undecided) | 0.0000 |
| potts/generic/reversed/uniform/K2 | 0.0107 | 0.0126 | yes | 0.0010 | 0.1315 | 0.0269 | K (decisive) | 0.0334 |
| potts/generic/reversed/uniform/K3 | 0.0108 | 0.0124 | yes | 0.0015 | 0.0273 | 0.0132 | K (undecided) | 0.0096 |
| potts/generic/reversed/uniform/K4 | 0.0115 | 0.0125 | yes | 0.0007 | 0.0136 | 0.0117 | K (undecided) | 0.0033 |
| potts/generic/reversed/uniform/K8 | 0.0110 | 0.0123 | yes | 0.0011 | 0.0110 | 0.0110 | K+1 (undecided) | 0.0002 |
| potts/square/forward/all_zero/K0 | 0.0000 | 0.0000 | yes | 0.0000 | n/a | 0.9699 | K (decisive) | 0.9781 |
| potts/square/forward/all_zero/K1 | 0.0088 | 0.0109 | yes | 0.0010 | 0.9699 | 0.2294 | K (decisive) | 0.2818 |
| potts/square/forward/all_zero/K16 | 0.0103 | 0.0125 | yes | 0.0013 | 0.0103 | 0.0103 | K-1 (undecided) | 0.0000 |
| potts/square/forward/all_zero/K2 | 0.0112 | 0.0122 | yes | 0.0014 | 0.2295 | 0.0460 | K (decisive) | 0.0602 |
| potts/square/forward/all_zero/K3 | 0.0109 | 0.0125 | yes | 0.0015 | 0.0461 | 0.0167 | K (undecided) | 0.0187 |
| potts/square/forward/all_zero/K4 | 0.0115 | 0.0124 | yes | 0.0027 | 0.0179 | 0.0121 | K (undecided) | 0.0069 |
| potts/square/forward/all_zero/K8 | 0.0108 | 0.0123 | yes | 0.0008 | 0.0108 | 0.0108 | K+1 (undecided) | 0.0005 |
| potts/square/forward/uniform/K0 | 0.0169 | 0.0185 | yes | 0.0015 | n/a | 0.6192 | K (decisive) | 0.6617 |
| potts/square/forward/uniform/K1 | 0.0113 | 0.0130 | yes | 0.0016 | 0.6192 | 0.1117 | K (decisive) | 0.1703 |
| potts/square/forward/uniform/K16 | 0.0113 | 0.0122 | yes | 0.0011 | 0.0113 | 0.0113 | K+1 (undecided) | 0.0000 |
| potts/square/forward/uniform/K2 | 0.0111 | 0.0127 | yes | 0.0009 | 0.1111 | 0.0402 | K (decisive) | 0.0701 |
| potts/square/forward/uniform/K3 | 0.0104 | 0.0124 | yes | 0.0012 | 0.0392 | 0.0218 | K (decisive) | 0.0352 |
| potts/square/forward/uniform/K4 | 0.0110 | 0.0123 | yes | 0.0013 | 0.0209 | 0.0153 | K (undecided) | 0.0192 |
| potts/square/forward/uniform/K8 | 0.0103 | 0.0125 | yes | 0.0012 | 0.0105 | 0.0104 | K (undecided) | 0.0018 |
| potts/square/reversed/all_zero/K0 | 0.0000 | 0.0000 | yes | 0.0000 | n/a | 0.9699 | K (decisive) | 0.9781 |
| potts/square/reversed/all_zero/K1 | 0.0087 | 0.0108 | yes | 0.0016 | 0.9693 | 0.2781 | K (decisive) | 0.3472 |
| potts/square/reversed/all_zero/K16 | 0.0106 | 0.0125 | yes | 0.0010 | 0.0106 | 0.0106 | K+1 (undecided) | 0.0000 |
| potts/square/reversed/all_zero/K2 | 0.0106 | 0.0123 | yes | 0.0010 | 0.2774 | 0.0594 | K (decisive) | 0.1056 |
| potts/square/reversed/all_zero/K3 | 0.0105 | 0.0125 | yes | 0.0008 | 0.0596 | 0.0284 | K (decisive) | 0.0551 |
| potts/square/reversed/all_zero/K4 | 0.0110 | 0.0124 | yes | 0.0016 | 0.0279 | 0.0189 | K (decisive) | 0.0302 |
| potts/square/reversed/all_zero/K8 | 0.0103 | 0.0124 | yes | 0.0010 | 0.0109 | 0.0105 | K (undecided) | 0.0029 |
| potts/square/reversed/uniform/K0 | 0.0163 | 0.0184 | yes | 0.0013 | n/a | 0.6163 | K (decisive) | 0.6617 |
| potts/square/reversed/uniform/K1 | 0.0117 | 0.0131 | yes | 0.0008 | 0.6164 | 0.1298 | K (decisive) | 0.1524 |
| potts/square/reversed/uniform/K16 | 0.0120 | 0.0124 | yes | 0.0022 | 0.0120 | 0.0120 | K+1 (undecided) | 0.0000 |
| potts/square/reversed/uniform/K2 | 0.0114 | 0.0126 | yes | 0.0014 | 0.1296 | 0.0277 | K (decisive) | 0.0334 |
| potts/square/reversed/uniform/K3 | 0.0105 | 0.0124 | yes | 0.0011 | 0.0278 | 0.0123 | K (undecided) | 0.0096 |
| potts/square/reversed/uniform/K4 | 0.0111 | 0.0125 | yes | 0.0015 | 0.0126 | 0.0115 | K (undecided) | 0.0033 |
| potts/square/reversed/uniform/K8 | 0.0112 | 0.0123 | yes | 0.0014 | 0.0112 | 0.0112 | K-1 (undecided) | 0.0002 |

`closest` names the exact law nearest to the histogram among p0 T^(K-1), p0 T^K and
p0 T^(K+1). It is `decisive` only when both neighbours sit more than the cell
tolerance away from p0 T^K on the exact side; otherwise the three laws are within
sampling noise of each other and the column carries no information.

## Negative controls

Exact-side gate on init `all_zero`: 32 checks, 0 without separation. Sampled rejections below.

| control | cell | exact separation | TV of samples to wrong ref | tolerance | rejected |
| --- | --- | --- | --- | --- | --- |
| negated_energy | potts/generic/forward/all_zero/K1 | 0.9872 | 0.9872 | 0.0109 | yes |
| negated_energy | potts/square/forward/all_zero/K1 | 0.9872 | 0.9871 | 0.0109 | yes |
| doubled_beta | potts/generic/forward/all_zero/K1 | 0.3377 | 0.3368 | 0.0109 | yes |
| doubled_beta | potts/square/forward/all_zero/K1 | 0.3377 | 0.3378 | 0.0109 | yes |
| transposed_w | potts/generic/forward/all_zero/K1 | 0.6678 | 0.6681 | 0.0109 | yes |
| transposed_w | potts/square/forward/all_zero/K1 | 0.6678 | 0.6681 | 0.0109 | yes |
| label_shift | potts/generic/forward/all_zero/K1 | 0.8133 | 0.8129 | 0.0109 | yes |
| label_shift | potts/square/forward/all_zero/K1 | 0.8133 | 0.8128 | 0.0109 | yes |
| reversed_order | potts/generic/forward/all_zero/K1 | 0.3481 | 0.3499 | 0.0109 | yes |
| reversed_order | potts/square/forward/all_zero/K1 | 0.3481 | 0.3483 | 0.0109 | yes |
| off_by_one | potts/generic/forward/all_zero/K1 | 0.2291 | 0.2302 | 0.0109 | yes |
| off_by_one | potts/square/forward/all_zero/K1 | 0.2291 | 0.2294 | 0.0109 | yes |
| negated_energy | potts/generic/forward/all_zero/K2 | 0.9611 | 0.9612 | 0.0122 | yes |
| negated_energy | potts/square/forward/all_zero/K2 | 0.9611 | 0.9614 | 0.0122 | yes |
| doubled_beta | potts/generic/forward/all_zero/K2 | 0.3634 | 0.3628 | 0.0122 | yes |
| doubled_beta | potts/square/forward/all_zero/K2 | 0.3634 | 0.3629 | 0.0122 | yes |
| transposed_w | potts/generic/forward/all_zero/K2 | 0.6214 | 0.6218 | 0.0122 | yes |
| transposed_w | potts/square/forward/all_zero/K2 | 0.6214 | 0.6219 | 0.0122 | yes |
| label_shift | potts/generic/forward/all_zero/K2 | 0.8134 | 0.8130 | 0.0122 | yes |
| label_shift | potts/square/forward/all_zero/K2 | 0.8134 | 0.8136 | 0.0122 | yes |
| reversed_order | potts/generic/forward/all_zero/K2 | 0.1001 | 0.1022 | 0.0122 | yes |
| reversed_order | potts/square/forward/all_zero/K2 | 0.1001 | 0.1001 | 0.0122 | yes |
| off_by_one | potts/generic/forward/all_zero/K2 | 0.0434 | 0.0456 | 0.0122 | yes |
| off_by_one | potts/square/forward/all_zero/K2 | 0.0434 | 0.0460 | 0.0122 | yes |
| negated_energy | potts/generic/forward/uniform/K1 | 0.9368 | 0.9369 | 0.0130 | yes |
| negated_energy | potts/square/forward/uniform/K1 | 0.9368 | 0.9367 | 0.0130 | yes |
| doubled_beta | potts/generic/forward/uniform/K1 | 0.3264 | 0.3267 | 0.0130 | yes |
| doubled_beta | potts/square/forward/uniform/K1 | 0.3264 | 0.3252 | 0.0130 | yes |
| transposed_w | potts/generic/forward/uniform/K1 | 0.5778 | 0.5792 | 0.0130 | yes |
| transposed_w | potts/square/forward/uniform/K1 | 0.5778 | 0.5775 | 0.0130 | yes |
| label_shift | potts/generic/forward/uniform/K1 | 0.7825 | 0.7819 | 0.0130 | yes |
| label_shift | potts/square/forward/uniform/K1 | 0.7825 | 0.7825 | 0.0130 | yes |
| reversed_order | potts/generic/forward/uniform/K1 | 0.1546 | 0.1553 | 0.0130 | yes |
| reversed_order | potts/square/forward/uniform/K1 | 0.1546 | 0.1565 | 0.0130 | yes |
| off_by_one | potts/generic/forward/uniform/K1 | 0.1106 | 0.1121 | 0.0130 | yes |
| off_by_one | potts/square/forward/uniform/K1 | 0.1106 | 0.1117 | 0.0130 | yes |
| negated_energy | potts/generic/forward/uniform/K2 | 0.9545 | 0.9547 | 0.0127 | yes |
| negated_energy | potts/square/forward/uniform/K2 | 0.9545 | 0.9545 | 0.0127 | yes |
| doubled_beta | potts/generic/forward/uniform/K2 | 0.3624 | 0.3618 | 0.0127 | yes |
| doubled_beta | potts/square/forward/uniform/K2 | 0.3624 | 0.3632 | 0.0127 | yes |
| transposed_w | potts/generic/forward/uniform/K2 | 0.6044 | 0.6043 | 0.0127 | yes |
| transposed_w | potts/square/forward/uniform/K2 | 0.6044 | 0.6048 | 0.0127 | yes |
| label_shift | potts/generic/forward/uniform/K2 | 0.8046 | 0.8045 | 0.0127 | yes |
| label_shift | potts/square/forward/uniform/K2 | 0.8046 | 0.8048 | 0.0127 | yes |
| reversed_order | potts/generic/forward/uniform/K2 | 0.0660 | 0.0671 | 0.0127 | yes |
| reversed_order | potts/square/forward/uniform/K2 | 0.0660 | 0.0662 | 0.0127 | yes |
| off_by_one | potts/generic/forward/uniform/K2 | 0.0382 | 0.0404 | 0.0127 | yes |
| off_by_one | potts/square/forward/uniform/K2 | 0.0382 | 0.0402 | 0.0127 | yes |
| negated_energy | potts/generic/reversed/all_zero/K1 | 0.9762 | 0.9762 | 0.0108 | yes |
| negated_energy | potts/square/reversed/all_zero/K1 | 0.9762 | 0.9765 | 0.0108 | yes |
| doubled_beta | potts/generic/reversed/all_zero/K1 | 0.3980 | 0.3976 | 0.0108 | yes |
| doubled_beta | potts/square/reversed/all_zero/K1 | 0.3980 | 0.3985 | 0.0108 | yes |
| transposed_w | potts/generic/reversed/all_zero/K1 | 0.7000 | 0.7003 | 0.0108 | yes |
| transposed_w | potts/square/reversed/all_zero/K1 | 0.7000 | 0.7005 | 0.0108 | yes |
| label_shift | potts/generic/reversed/all_zero/K1 | 0.8102 | 0.8106 | 0.0108 | yes |
| label_shift | potts/square/reversed/all_zero/K1 | 0.8102 | 0.8105 | 0.0108 | yes |
| reversed_order | potts/generic/reversed/all_zero/K1 | 0.3481 | 0.3482 | 0.0108 | yes |
| reversed_order | potts/square/reversed/all_zero/K1 | 0.3481 | 0.3477 | 0.0108 | yes |
| off_by_one | potts/generic/reversed/all_zero/K1 | 0.2783 | 0.2778 | 0.0108 | yes |
| off_by_one | potts/square/reversed/all_zero/K1 | 0.2783 | 0.2781 | 0.0108 | yes |
| negated_energy | potts/generic/reversed/all_zero/K2 | 0.9556 | 0.9559 | 0.0123 | yes |
| negated_energy | potts/square/reversed/all_zero/K2 | 0.9556 | 0.9561 | 0.0123 | yes |
| doubled_beta | potts/generic/reversed/all_zero/K2 | 0.3826 | 0.3814 | 0.0123 | yes |
| doubled_beta | potts/square/reversed/all_zero/K2 | 0.3826 | 0.3815 | 0.0123 | yes |
| transposed_w | potts/generic/reversed/all_zero/K2 | 0.6207 | 0.6218 | 0.0123 | yes |
| transposed_w | potts/square/reversed/all_zero/K2 | 0.6207 | 0.6222 | 0.0123 | yes |
| label_shift | potts/generic/reversed/all_zero/K2 | 0.7775 | 0.7779 | 0.0123 | yes |
| label_shift | potts/square/reversed/all_zero/K2 | 0.7775 | 0.7780 | 0.0123 | yes |
| reversed_order | potts/generic/reversed/all_zero/K2 | 0.1001 | 0.1020 | 0.0123 | yes |
| reversed_order | potts/square/reversed/all_zero/K2 | 0.1001 | 0.1015 | 0.0123 | yes |
| off_by_one | potts/generic/reversed/all_zero/K2 | 0.0575 | 0.0608 | 0.0123 | yes |
| off_by_one | potts/square/reversed/all_zero/K2 | 0.0575 | 0.0594 | 0.0123 | yes |
| negated_energy | potts/generic/reversed/uniform/K1 | 0.9355 | 0.9354 | 0.0131 | yes |
| negated_energy | potts/square/reversed/uniform/K1 | 0.9355 | 0.9357 | 0.0131 | yes |
| doubled_beta | potts/generic/reversed/uniform/K1 | 0.3098 | 0.3097 | 0.0131 | yes |
| doubled_beta | potts/square/reversed/uniform/K1 | 0.3098 | 0.3090 | 0.0131 | yes |
| transposed_w | potts/generic/reversed/uniform/K1 | 0.5761 | 0.5745 | 0.0131 | yes |
| transposed_w | potts/square/reversed/uniform/K1 | 0.5761 | 0.5764 | 0.0131 | yes |
| label_shift | potts/generic/reversed/uniform/K1 | 0.7878 | 0.7874 | 0.0131 | yes |
| label_shift | potts/square/reversed/uniform/K1 | 0.7878 | 0.7870 | 0.0131 | yes |
| reversed_order | potts/generic/reversed/uniform/K1 | 0.1546 | 0.1558 | 0.0131 | yes |
| reversed_order | potts/square/reversed/uniform/K1 | 0.1546 | 0.1558 | 0.0131 | yes |
| off_by_one | potts/generic/reversed/uniform/K1 | 0.1293 | 0.1286 | 0.0131 | yes |
| off_by_one | potts/square/reversed/uniform/K1 | 0.1293 | 0.1298 | 0.0131 | yes |
| negated_energy | potts/generic/reversed/uniform/K2 | 0.9534 | 0.9533 | 0.0126 | yes |
| negated_energy | potts/square/reversed/uniform/K2 | 0.9534 | 0.9532 | 0.0126 | yes |
| doubled_beta | potts/generic/reversed/uniform/K2 | 0.3494 | 0.3479 | 0.0126 | yes |
| doubled_beta | potts/square/reversed/uniform/K2 | 0.3494 | 0.3497 | 0.0126 | yes |
| transposed_w | potts/generic/reversed/uniform/K2 | 0.5997 | 0.5992 | 0.0126 | yes |
| transposed_w | potts/square/reversed/uniform/K2 | 0.5997 | 0.5998 | 0.0126 | yes |
| label_shift | potts/generic/reversed/uniform/K2 | 0.8045 | 0.8038 | 0.0126 | yes |
| label_shift | potts/square/reversed/uniform/K2 | 0.8045 | 0.8042 | 0.0126 | yes |
| reversed_order | potts/generic/reversed/uniform/K2 | 0.0660 | 0.0680 | 0.0126 | yes |
| reversed_order | potts/square/reversed/uniform/K2 | 0.0660 | 0.0684 | 0.0126 | yes |
| off_by_one | potts/generic/reversed/uniform/K2 | 0.0242 | 0.0269 | 0.0126 | yes |
| off_by_one | potts/square/reversed/uniform/K2 | 0.0242 | 0.0277 | 0.0126 | yes |
| clamp_as_zero | clamped/generic/forward/all_zero/K1 | 0.3823 | 0.3823 | 0.0080 | yes |
| clamp_as_zero | clamped/square/forward/all_zero/K1 | 0.3823 | 0.3820 | 0.0080 | yes |
| clamp_as_zero | clamped/generic/forward/all_zero/K2 | 0.4816 | 0.4811 | 0.0086 | yes |
| clamp_as_zero | clamped/square/forward/all_zero/K2 | 0.4816 | 0.4820 | 0.0086 | yes |
| doubled_beta | bridge/generic/forward/all_zero/K1 | 0.1732 | 0.1726 | 0.0050 | yes |
| doubled_beta | bridge/square/forward/all_zero/K1 | 0.1732 | 0.1734 | 0.0050 | yes |
| label_swap | bridge/generic/forward/all_zero/K1 | 0.2201 | 0.2204 | 0.0050 | yes |
| label_swap | bridge/square/forward/all_zero/K1 | 0.2201 | 0.2205 | 0.0050 | yes |
| off_by_one | bridge/generic/forward/all_zero/K1 | 0.1285 | 0.1289 | 0.0050 | yes |
| off_by_one | bridge/square/forward/all_zero/K1 | 0.1285 | 0.1284 | 0.0050 | yes |
| doubled_beta | bridge/generic/forward/all_zero/K2 | 0.1700 | 0.1701 | 0.0049 | yes |
| doubled_beta | bridge/square/forward/all_zero/K2 | 0.1700 | 0.1708 | 0.0049 | yes |
| label_swap | bridge/generic/forward/all_zero/K2 | 0.1548 | 0.1546 | 0.0049 | yes |
| label_swap | bridge/square/forward/all_zero/K2 | 0.1548 | 0.1551 | 0.0049 | yes |
| off_by_one | bridge/generic/forward/all_zero/K2 | 0.0063 | 0.0073 | 0.0049 | yes |
| off_by_one | bridge/square/forward/all_zero/K2 | 0.0063 | 0.0069 | 0.0049 | yes |
