# NumCompute benchmark results

## Environment

- python: 3.14.0
- numpy: 2.4.0
- platform: Windows-11-10.0.26300-SP0
- machine: AMD64
- processor: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
- cpu_count: 20
- date: 2026-10-20 12:44 UTC
- seed: 0
- repeats: 3 (best time reported)
- mode: full

## Standardise columns (10 features)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 1,000 | 4.70 ms | 161.8 us | 29.0x | yes |
| 5,000 | 22.47 ms | 750.5 us | 29.9x | yes |
| 20,000 | 92.30 ms | 2.80 ms | 33.0x | yes |

## Row softmax (20 classes)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 1,000 | 2.30 ms | 166.5 us | 13.8x | yes |
| 5,000 | 16.09 ms | 1.44 ms | 11.2x | yes |
| 20,000 | 74.41 ms | 5.24 ms | 14.2x | yes |

## Pairwise Euclidean distances (8 features)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 100 | 12.08 ms | 43.5 us | 277.8x | yes |
| 200 | 48.51 ms | 344.8 us | 140.7x | yes |
| 400 | 209.58 ms | 1.77 ms | 118.2x | yes |

## Average ranks with ties

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 500 | 10.36 ms | 60.7 us | 170.7x | yes |
| 1,000 | 40.07 ms | 87.3 us | 459.0x | yes |
| 2,000 | 164.44 ms | 176.1 us | 933.8x | yes |

## Top-10 of n values

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 10,000 | 2.36 ms | 133.4 us | 17.7x | yes |
| 100,000 | 42.38 ms | 1.27 ms | 33.3x | yes |
| 300,000 | 194.32 ms | 3.46 ms | 56.2x | yes |

## Histogram (20 bins)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 10,000 | 999.5 us | 172.1 us | 5.8x | yes |
| 100,000 | 11.31 ms | 1.94 ms | 5.8x | yes |
| 500,000 | 63.22 ms | 10.33 ms | 6.1x | yes |

## Streaming mean/variance (Welford loop vs batch)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 10,000 | 779.4 us | 103.4 us | 7.5x | yes |
| 100,000 | 7.24 ms | 836.2 us | 8.7x | yes |
| 500,000 | 37.98 ms | 4.08 ms | 9.3x | yes |

## Finite-difference gradient (one call per point vs one batched call)

| Size | Loop | Vectorised | Speed-up | Match |
|---:|---:|---:|---:|:---:|
| 100 | 550.2 us | 32.3 us | 17.0x | yes |
| 500 | 4.07 ms | 2.23 ms | 1.8x | yes |
| 1,500 | 25.76 ms | 24.43 ms | 1.1x | yes |
