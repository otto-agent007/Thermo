# THRML finite-sweep contract (E0 / A1)

Exact references: float64 enumeration of the five-spin chain and powers of the
32x32 ordered block-Gibbs sweep matrix (`exact_reference`). THRML cells: 0.1.4 on
CPU, float32, 400000 independent chains per cell (`software_simulation`). No hardware
evidence anywhere in this report.

Request digest `sha256:fd9ebdbf2435fa46a5f8fcf7f67045321e843f49df32802f9dd5f50860890e36`; result digest `sha256:46d6812ab5d0bcc70cfe2a105e7823f5255a65742b5b0ca8f9252750ef952d37`.

## Cells

| cell | TV to p0 T^K | tolerance (q=0.999) | pass | TV to p0 T^(K-1) | TV to p0 T^(K+1) | closest | TV to stationary (exact) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clamped+/forward/hinton/K0 | 0.0021 | 0.0040 | yes | n/a | 0.1681 | K (decisive) | 0.1673 |
| clamped+/forward/hinton/K1 | 0.0021 | 0.0040 | yes | 0.1678 | 0.0132 | K (decisive) | 0.0154 |
| clamped+/forward/hinton/K16 | 0.0025 | 0.0039 | yes | 0.0025 | 0.0025 | K-1 (undecided) | 0.0000 |
| clamped+/forward/hinton/K2 | 0.0019 | 0.0040 | yes | 0.0135 | 0.0023 | K (undecided) | 0.0013 |
| clamped+/forward/hinton/K3 | 0.0016 | 0.0041 | yes | 0.0019 | 0.0017 | K (undecided) | 0.0001 |
| clamped+/forward/hinton/K30 | 0.0016 | 0.0040 | yes | 0.0016 | 0.0016 | K-1 (undecided) | 0.0000 |
| clamped+/forward/hinton/K4 | 0.0030 | 0.0040 | yes | 0.0030 | 0.0030 | K+1 (undecided) | 0.0000 |
| clamped+/forward/hinton/K8 | 0.0021 | 0.0041 | yes | 0.0021 | 0.0021 | K+1 (undecided) | 0.0000 |
| clamped-/forward/hinton/K0 | 0.0021 | 0.0040 | yes | n/a | 0.1933 | K (decisive) | 0.1988 |
| clamped-/forward/hinton/K1 | 0.0023 | 0.0039 | yes | 0.1925 | 0.0329 | K (decisive) | 0.0375 |
| clamped-/forward/hinton/K16 | 0.0032 | 0.0038 | yes | 0.0032 | 0.0032 | K+1 (undecided) | 0.0000 |
| clamped-/forward/hinton/K2 | 0.0025 | 0.0040 | yes | 0.0361 | 0.0031 | K (undecided) | 0.0033 |
| clamped-/forward/hinton/K3 | 0.0033 | 0.0041 | yes | 0.0026 | 0.0035 | K-1 (undecided) | 0.0003 |
| clamped-/forward/hinton/K30 | 0.0027 | 0.0040 | yes | 0.0027 | 0.0027 | K+1 (undecided) | 0.0000 |
| clamped-/forward/hinton/K4 | 0.0021 | 0.0041 | yes | 0.0021 | 0.0021 | K+1 (undecided) | 0.0000 |
| clamped-/forward/hinton/K8 | 0.0031 | 0.0039 | yes | 0.0031 | 0.0031 | K+1 (undecided) | 0.0000 |
| free/forward/all_minus/K0 | 0.0000 | 0.0000 | yes | n/a | 0.9343 | K (decisive) | 0.9603 |
| free/forward/all_minus/K1 | 0.0035 | 0.0049 | yes | 0.9346 | 0.1291 | K (decisive) | 0.1344 |
| free/forward/all_minus/K16 | 0.0037 | 0.0051 | yes | 0.0037 | 0.0037 | K-1 (undecided) | 0.0000 |
| free/forward/all_minus/K2 | 0.0026 | 0.0050 | yes | 0.1283 | 0.0064 | K (decisive) | 0.0067 |
| free/forward/all_minus/K3 | 0.0041 | 0.0049 | yes | 0.0070 | 0.0042 | K (undecided) | 0.0005 |
| free/forward/all_minus/K30 | 0.0033 | 0.0049 | yes | 0.0033 | 0.0033 | K (undecided) | 0.0000 |
| free/forward/all_minus/K4 | 0.0036 | 0.0048 | yes | 0.0037 | 0.0036 | K (undecided) | 0.0000 |
| free/forward/all_minus/K8 | 0.0040 | 0.0049 | yes | 0.0040 | 0.0040 | K-1 (undecided) | 0.0000 |
| free/forward/hinton/K0 | 0.0041 | 0.0051 | yes | n/a | 0.1842 | K (decisive) | 0.1843 |
| free/forward/hinton/K1 | 0.0030 | 0.0049 | yes | 0.1843 | 0.0080 | K (decisive) | 0.0078 |
| free/forward/hinton/K16 | 0.0035 | 0.0048 | yes | 0.0035 | 0.0035 | K-1 (undecided) | 0.0000 |
| free/forward/hinton/K2 | 0.0036 | 0.0049 | yes | 0.0080 | 0.0034 | K+1 (undecided) | 0.0010 |
| free/forward/hinton/K3 | 0.0031 | 0.0050 | yes | 0.0029 | 0.0031 | K-1 (undecided) | 0.0001 |
| free/forward/hinton/K30 | 0.0032 | 0.0050 | yes | 0.0032 | 0.0032 | K+1 (undecided) | 0.0000 |
| free/forward/hinton/K4 | 0.0039 | 0.0050 | yes | 0.0040 | 0.0039 | K+1 (undecided) | 0.0000 |
| free/forward/hinton/K8 | 0.0036 | 0.0049 | yes | 0.0036 | 0.0036 | K-1 (undecided) | 0.0000 |
| free/forward/uniform/K0 | 0.0039 | 0.0051 | yes | n/a | 0.2029 | K (decisive) | 0.2006 |
| free/forward/uniform/K1 | 0.0034 | 0.0050 | yes | 0.2015 | 0.0162 | K (decisive) | 0.0179 |
| free/forward/uniform/K16 | 0.0030 | 0.0050 | yes | 0.0030 | 0.0030 | K-1 (undecided) | 0.0000 |
| free/forward/uniform/K2 | 0.0038 | 0.0051 | yes | 0.0168 | 0.0042 | K (undecided) | 0.0024 |
| free/forward/uniform/K3 | 0.0033 | 0.0050 | yes | 0.0031 | 0.0033 | K-1 (undecided) | 0.0003 |
| free/forward/uniform/K30 | 0.0034 | 0.0050 | yes | 0.0034 | 0.0034 | K+1 (undecided) | 0.0000 |
| free/forward/uniform/K4 | 0.0034 | 0.0050 | yes | 0.0035 | 0.0034 | K+1 (undecided) | 0.0000 |
| free/forward/uniform/K8 | 0.0032 | 0.0050 | yes | 0.0032 | 0.0032 | K+1 (undecided) | 0.0000 |
| free/reversed/all_minus/K0 | 0.0000 | 0.0000 | yes | n/a | 0.9343 | K (decisive) | 0.9603 |
| free/reversed/all_minus/K1 | 0.0024 | 0.0049 | yes | 0.9338 | 0.1855 | K (decisive) | 0.1980 |
| free/reversed/all_minus/K16 | 0.0033 | 0.0050 | yes | 0.0033 | 0.0033 | K+1 (undecided) | 0.0000 |
| free/reversed/all_minus/K2 | 0.0031 | 0.0051 | yes | 0.1848 | 0.0121 | K (decisive) | 0.0125 |
| free/reversed/all_minus/K3 | 0.0029 | 0.0051 | yes | 0.0121 | 0.0032 | K (undecided) | 0.0011 |
| free/reversed/all_minus/K30 | 0.0035 | 0.0050 | yes | 0.0035 | 0.0035 | K+1 (undecided) | 0.0000 |
| free/reversed/all_minus/K4 | 0.0037 | 0.0049 | yes | 0.0039 | 0.0037 | K+1 (undecided) | 0.0001 |
| free/reversed/all_minus/K8 | 0.0034 | 0.0051 | yes | 0.0034 | 0.0034 | K-1 (undecided) | 0.0000 |
| free/reversed/hinton/K0 | 0.0039 | 0.0051 | yes | n/a | 0.1843 | K (decisive) | 0.1843 |
| free/reversed/hinton/K1 | 0.0034 | 0.0050 | yes | 0.1826 | 0.0062 | K (decisive) | 0.0055 |
| free/reversed/hinton/K16 | 0.0029 | 0.0049 | yes | 0.0029 | 0.0029 | K+1 (undecided) | 0.0000 |
| free/reversed/hinton/K2 | 0.0032 | 0.0050 | yes | 0.0065 | 0.0031 | K+1 (undecided) | 0.0004 |
| free/reversed/hinton/K3 | 0.0029 | 0.0049 | yes | 0.0029 | 0.0029 | K (undecided) | 0.0000 |
| free/reversed/hinton/K30 | 0.0034 | 0.0049 | yes | 0.0034 | 0.0034 | K-1 (undecided) | 0.0000 |
| free/reversed/hinton/K4 | 0.0033 | 0.0050 | yes | 0.0033 | 0.0033 | K+1 (undecided) | 0.0000 |
| free/reversed/hinton/K8 | 0.0031 | 0.0048 | yes | 0.0031 | 0.0031 | K-1 (undecided) | 0.0000 |
| free/reversed/uniform/K0 | 0.0036 | 0.0051 | yes | n/a | 0.2002 | K (decisive) | 0.2006 |
| free/reversed/uniform/K1 | 0.0029 | 0.0051 | yes | 0.2001 | 0.0102 | K (decisive) | 0.0098 |
| free/reversed/uniform/K16 | 0.0034 | 0.0050 | yes | 0.0034 | 0.0034 | K-1 (undecided) | 0.0000 |
| free/reversed/uniform/K2 | 0.0042 | 0.0052 | yes | 0.0089 | 0.0044 | K (undecided) | 0.0008 |
| free/reversed/uniform/K3 | 0.0030 | 0.0050 | yes | 0.0029 | 0.0030 | K-1 (undecided) | 0.0001 |
| free/reversed/uniform/K30 | 0.0034 | 0.0049 | yes | 0.0034 | 0.0034 | K-1 (undecided) | 0.0000 |
| free/reversed/uniform/K4 | 0.0032 | 0.0050 | yes | 0.0032 | 0.0032 | K-1 (undecided) | 0.0000 |
| free/reversed/uniform/K8 | 0.0035 | 0.0050 | yes | 0.0035 | 0.0035 | K+1 (undecided) | 0.0000 |

`closest` names the exact law nearest to the histogram among p0 T^(K-1), p0 T^K and
p0 T^(K+1). It is `decisive` only when both neighbours sit more than the cell
tolerance away from p0 T^K on the exact side; otherwise the three laws are within
sampling noise of each other and the column carries no information.

## Negative controls

Exact-side gate on init `all_minus`: 16 checks, 0 without separation. Sampled rejections below.

| control | cell | exact separation | TV of samples to wrong ref | tolerance | rejected |
| --- | --- | --- | --- | --- | --- |
| flipped_j | free/forward/all_minus/K1 | 0.3850 | 0.3853 | 0.0049 | yes |
| inverse_beta | free/forward/all_minus/K1 | 0.1023 | 0.1024 | 0.0049 | yes |
| reversed_order | free/forward/all_minus/K1 | 0.1960 | 0.1952 | 0.0049 | yes |
| off_by_one | free/forward/all_minus/K1 | 0.1285 | 0.1291 | 0.0049 | yes |
| flipped_j | free/forward/all_minus/K2 | 0.3469 | 0.3468 | 0.0050 | yes |
| inverse_beta | free/forward/all_minus/K2 | 0.1019 | 0.1018 | 0.0050 | yes |
| reversed_order | free/forward/all_minus/K2 | 0.0121 | 0.0128 | 0.0050 | yes |
| off_by_one | free/forward/all_minus/K2 | 0.0063 | 0.0064 | 0.0050 | yes |
| flipped_j | free/forward/uniform/K1 | 0.3424 | 0.3417 | 0.0050 | yes |
| inverse_beta | free/forward/uniform/K1 | 0.1000 | 0.1006 | 0.0050 | yes |
| reversed_order | free/forward/uniform/K1 | 0.0216 | 0.0218 | 0.0050 | yes |
| off_by_one | free/forward/uniform/K1 | 0.0156 | 0.0162 | 0.0050 | yes |
| flipped_j | free/forward/uniform/K2 | 0.3459 | 0.3450 | 0.0051 | yes |
| inverse_beta | free/forward/uniform/K2 | 0.1032 | 0.1038 | 0.0051 | yes |
| reversed_order | free/forward/uniform/K2 | 0.0027 | 0.0046 | 0.0051 | NO |
| off_by_one | free/forward/uniform/K2 | 0.0021 | 0.0042 | 0.0051 | NO |
| flipped_j | free/forward/hinton/K1 | 0.3438 | 0.3437 | 0.0049 | yes |
| inverse_beta | free/forward/hinton/K1 | 0.1008 | 0.1009 | 0.0049 | yes |
| reversed_order | free/forward/hinton/K1 | 0.0081 | 0.0089 | 0.0049 | yes |
| off_by_one | free/forward/hinton/K1 | 0.0068 | 0.0080 | 0.0049 | yes |
| flipped_j | free/forward/hinton/K2 | 0.3461 | 0.3471 | 0.0049 | yes |
| inverse_beta | free/forward/hinton/K2 | 0.1036 | 0.1021 | 0.0049 | yes |
| reversed_order | free/forward/hinton/K2 | 0.0010 | 0.0035 | 0.0049 | NO |
| off_by_one | free/forward/hinton/K2 | 0.0008 | 0.0034 | 0.0049 | NO |
| flipped_j | free/reversed/all_minus/K1 | 0.4616 | 0.4616 | 0.0049 | yes |
| inverse_beta | free/reversed/all_minus/K1 | 0.1193 | 0.1194 | 0.0049 | yes |
| reversed_order | free/reversed/all_minus/K1 | 0.1960 | 0.1963 | 0.0049 | yes |
| off_by_one | free/reversed/all_minus/K1 | 0.1854 | 0.1855 | 0.0049 | yes |
| flipped_j | free/reversed/all_minus/K2 | 0.3477 | 0.3485 | 0.0051 | yes |
| inverse_beta | free/reversed/all_minus/K2 | 0.0975 | 0.0974 | 0.0051 | yes |
| reversed_order | free/reversed/all_minus/K2 | 0.0121 | 0.0130 | 0.0051 | yes |
| off_by_one | free/reversed/all_minus/K2 | 0.0115 | 0.0121 | 0.0051 | yes |
| flipped_j | free/reversed/uniform/K1 | 0.3463 | 0.3475 | 0.0051 | yes |
| inverse_beta | free/reversed/uniform/K1 | 0.0994 | 0.0991 | 0.0051 | yes |
| reversed_order | free/reversed/uniform/K1 | 0.0216 | 0.0227 | 0.0051 | yes |
| off_by_one | free/reversed/uniform/K1 | 0.0090 | 0.0102 | 0.0051 | yes |
| flipped_j | free/reversed/uniform/K2 | 0.3464 | 0.3468 | 0.0052 | yes |
| inverse_beta | free/reversed/uniform/K2 | 0.1041 | 0.1036 | 0.0052 | yes |
| reversed_order | free/reversed/uniform/K2 | 0.0027 | 0.0056 | 0.0052 | yes |
| off_by_one | free/reversed/uniform/K2 | 0.0007 | 0.0044 | 0.0052 | NO |
| flipped_j | free/reversed/hinton/K1 | 0.3462 | 0.3461 | 0.0050 | yes |
| inverse_beta | free/reversed/hinton/K1 | 0.0999 | 0.1000 | 0.0050 | yes |
| reversed_order | free/reversed/hinton/K1 | 0.0081 | 0.0093 | 0.0050 | yes |
| off_by_one | free/reversed/hinton/K1 | 0.0051 | 0.0062 | 0.0050 | yes |
| flipped_j | free/reversed/hinton/K2 | 0.3464 | 0.3463 | 0.0050 | yes |
| inverse_beta | free/reversed/hinton/K2 | 0.1040 | 0.1040 | 0.0050 | yes |
| reversed_order | free/reversed/hinton/K2 | 0.0010 | 0.0033 | 0.0050 | NO |
| off_by_one | free/reversed/hinton/K2 | 0.0003 | 0.0031 | 0.0050 | NO |

## Initializer encoding (K=0, hinton_init)

TV to product sigmoid(beta h): 0.0041; TV to product sigmoid(2 beta h): 0.0449; exact separation between the two hypotheses: 0.0454.
