# 03 Activation Steering Brief

**Station role:** Security Sentinel
**Mission 4 objective [4]:** extract contrast activation vectors and implement inference-time activation steering to suppress malicious personas.

## 1. Idea (Representation Engineering, Zou et al. 2023)
Training-based alignment (SFT, DPO) changes the weights. Steering changes only the activations while the model runs, so it can be switched on, tuned and removed without retraining.

1. **Contrast:** collect activations at one layer for persona examples (malicious) and contrast examples (benign).
2. **Vector:** v = mean(h_malicious) - mean(h_benign), scaled to unit length.
3. **Steer:** register a forward hook that sets h' = h + alpha * v. To suppress the persona, use a negative alpha.

## 2. Files
| File | Purpose |
|---|---|
| `activation_steering.py` | `last_token_activations` (captures a layer output), `contrast_vector`, `ActivationSteeringHook` (attach / remove / set_alpha / context manager, supports tensor and tuple layer outputs), `mass_on_tokens` (probability mass on a token set). |
| `tests/test_steering.py` | 7 PyTorch tests: hook adds alpha*v, remove restores output, weights unchanged, alpha 0 is identity, contrast vector direction and shape check, tuple outputs (skipped if torch is missing). |

## 3. Safety checks built in
- The hook never modifies parameters; a test compares weights before and after.
- `remove()` restores the original forward behaviour; hooks are always removed in `finally` or by the context manager.
- `contrast_vector` raises an error if the two activation matrices differ in width. The hook itself does not check shapes; PyTorch raises its own error if the vector does not fit the layer width.

## 4. Risks to report
- A large |alpha| degrades fluency. The steering sweep prints benign-token mass next to malicious-token mass so degradation is visible.
- The best layer and alpha must be found by experiment; mid-depth layers are a common start, not a rule.
- Steering is a mitigation, not a guarantee: a determined prompt can still elicit the behaviour, and the vector only captures the contrast present in the examples used.
- The demo persona is a toy token-set split. On a real model, build the contrast set from real labelled prompts and test on held-out ones.

## 5. How to run
```
python ps04_engine.py --demo steer     (needs torch)
```
