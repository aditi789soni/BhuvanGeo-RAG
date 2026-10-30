# BhuvanGeo-RAG Performance & Verification Benchmark

- **Total Queries Evaluated:** 13
- **Verification / Anti-Hallucination Accuracy:** 100.0%
- **Average Retrieval Latency:** 116.48 ms
- **Average Answering Latency:** 81.47 ms
- **Average End-to-End Latency:** 197.95 ms

## Test Results Table

| ID | Category | Retrieval (ms) | Answering (ms) | Total (ms) | Grounded | Verification |
|---|---|---|---|---|---|---|
| `Q1_LISS4_SPECS` | Supported - Payload Specification | 267.8 | 232.67 | 500.47 | True | ✅ Pass |
| `Q2_LISS3_BANDS` | Supported - Spectral Bands | 109.63 | 65.37 | 175.0 | True | ✅ Pass |
| `Q3_AWIFS_CHARACTERISTICS` | Supported - Payload Specification | 102.36 | 229.05 | 331.41 | True | ✅ Pass |
| `Q4_CARTOSAT1_STEREO` | Supported - Cartography & DEM | 87.03 | 54.71 | 141.74 | True | ✅ Pass |
| `Q5_CARTOSAT2_SUBMETER` | Supported - High Resolution Imaging | 61.95 | 62.88 | 124.83 | True | ✅ Pass |
| `Q6_OCM_OCEAN_COLOUR` | Supported - Oceanography | 219.8 | 44.23 | 264.03 | True | ✅ Pass |
| `Q7_OSCAT_SCATTEROMETER` | Supported - Radar Meteorology | 41.5 | 66.22 | 107.72 | True | ✅ Pass |
| `Q8_CADASTRAL_SUITABILITY` | Supported - Geospatial Application | 44.14 | 166.66 | 210.8 | True | ✅ Pass |
| `Q9_AWIFS_REVISIT` | Supported - Temporal Resolution | 97.07 | 45.9 | 142.97 | True | ✅ Pass |
| `Q10_RADIOMETRIC_BITS` | Supported - Radiometry | 48.98 | 91.43 | 140.41 | True | ✅ Pass |
| `Q11_CHANDRAYAAN_ROVER` | Anti-Hallucination Probe | 311.8 | 0.0 | 311.8 | False | ✅ Pass |
| `Q12_JWST_MIRROR` | Anti-Hallucination Probe | 76.78 | 0.0 | 76.78 | False | ✅ Pass |
| `Q13_MARS_ROVER_SPEED` | Anti-Hallucination Probe | 45.39 | 0.0 | 45.39 | False | ✅ Pass |
