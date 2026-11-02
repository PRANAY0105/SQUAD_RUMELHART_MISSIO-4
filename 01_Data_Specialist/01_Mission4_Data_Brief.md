# 01 Mission 4 Data Brief

**Station role:** Data Specialist
**Mission 4 objective [3]:** build a length-bias diagnostic test suite and implement the SimPO length-normalised, reference-free loss.

## 1. The problem this role solves
Standard DPO sums log-probabilities over every token. A longer answer has more terms in the sum, so a model can raise its reward margin by becoming wordy. Humans often rate longer answers higher, so the data pushes the same way. The Data Specialist measures that bias and supplies the fix.

## 2. Files
| File | Purpose |
|---|---|
| `simpo_loss.py` | `SimPOLoss(beta, gamma)`: reward r = (beta/|y|) log pi(y|x), loss = -log sigmoid(r_w - r_l - gamma). No reference model. |
| `length_bias_diagnostics.py` | numpy diagnostics: correlation of reward margin with length difference, accuracy when the chosen answer is longer versus shorter, and a verbosity-exploit score. |
| `tests/test_length_bias.py` | 4 numpy tests (run anywhere) and 3 PyTorch tests for SimPO (skipped if torch is missing). |

## 3. How to read the diagnostic
- **Exploit score** = accuracy on pairs where the chosen answer is longer minus accuracy on pairs where it is shorter. Near 0 is healthy. Above 0.2 (the default threshold, an arbitrary choice you can change) is flagged as length bias.
- The test set must contain both kinds of pairs. Otherwise the report says INCONCLUSIVE instead of guessing.

## 4. SimPO versus DPO in one table
| | DPO | SimPO |
|---|---|---|
| Reward | beta log(pi/pi_ref), summed over tokens | (beta/|y|) log pi, averaged over tokens |
| Reference model in memory | yes | no |
| Length handling | none | normalised by |y| |
| Extra hyper-parameter | none | margin gamma |
| Typical beta | about 0.1 | about 2 to 2.5 |

## 5. How to run
```
python run_tests.py            (from 04_Systems_Architect; runs all roles)
python ps04_engine.py --demo dpo   (needs torch; writes the DPO vs SimPO length-bias table)
```

## 6. Honest limits
- The benchmark uses a toy model and one random seed. It demonstrates the procedure; it does not prove SimPO beats DPO on real LLMs.
- No real preference dataset was used. Replace the toy pairs with real (prompt, chosen, rejected) data to audit a real model.
- Length is counted in tokens of the toy vocabulary; with a real tokenizer count tokens the same way for both answers.
