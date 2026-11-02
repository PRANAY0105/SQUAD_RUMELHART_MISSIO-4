# Activation steering sweep

| alpha (negative = steer away from malicious persona) | P(malicious tokens) | P(benign tokens) |
|---|---|---|
| 0.0 | 0.9938 | 0.0055 |
| -2.0 | 0.9793 | 0.0193 |
| -5.0 | 0.6733 | 0.3224 |
| -10.0 | 0.0113 | 0.9881 |
| -20.0 | 0.0000 | 0.9999 |

Context: malicious-persona prompts. Weights unchanged after steering: True. Large |alpha| can push the toy model off-distribution; check that fluency (benign mass) is still sensible.
