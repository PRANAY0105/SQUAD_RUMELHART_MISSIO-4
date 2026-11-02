# 04 Systems Architect Brief

**Station role:** Systems Architect
**Mission 4 objective [1]:** implement the mathematical DPO loss and the Bradley-Terry log-ratio computation from scratch in PyTorch. Also owns integration (objective [5]).

## 1. The derivation in four lines
1. KL-regularised RLHF has the optimal policy pi*(y|x) = (1/Z(x)) pi_ref(y|x) exp(r(x,y)/beta).
2. Invert it: r(x,y) = beta log(pi*/pi_ref) + beta log Z(x).
3. Put that into Bradley-Terry, P(y_w > y_l) = sigmoid(r_w - r_l). The two log Z(x) terms cancel.
4. Maximise the likelihood: L_DPO = -E log sigmoid( beta[ log(pi_theta(y_w)/pi_ref(y_w)) - log(pi_theta(y_l)/pi_ref(y_l)) ] ).

Gradient weight: beta * sigmoid(r_hat_l - r_hat_w). High when the model ranks the loser above the winner, near 0 when it is already confident.

## 2. Files
| File | Purpose |
|---|---|
| `dpo_loss.py` | `sequence_logprobs`, `log_ratios`, `dpo_loss` (with optional label smoothing, returns loss, rewards and metrics), `dpo_gradient_weight`. |
| `toy_lm.py` | Small causal LM, SFT pre-training and synthetic preference pairs so the whole mission runs on a CPU. |
| `ps04_engine.py` | Runs the demos: DPO vs SimPO length-bias benchmark, pair comparison, steering sweep, alignment-tax template. Writes `outputs/*.md`. |
| `alignment_tax.py` + `example_scores.json` | Alignment tax audit: per-benchmark delta and relative tax against a tolerance. The JSON is a placeholder template. |
| `ps04_solution.ipynb` | Integrated notebook for the whole squad with sign-off table. |
| `verify_math_numpy.py` | Torch-free check of the maths: ln 2 at zero margin, gradient against finite differences, log Z cancellation, length simulation. |
| `run_tests.py`, `tests/` | Runs the tests of all four roles. |
| `.vscode/launch.json`, `requirements.txt`, `.gitignore` | Run configurations and dependencies. |

## 3. Integration order
1. `python verify_math_numpy.py` (no torch needed).
2. `pip install torch numpy`, then `python run_tests.py` (all 26 tests should run, none skipped).
3. `python ps04_engine.py --demo all`, or open `ps04_solution.ipynb` and Run All.
4. Fill `example_scores.json` with measured benchmark numbers and run `python alignment_tax.py`.
5. Create the GitHub repository from your own account and paste the URL into the notebook sign-off.

## 4. Alignment tax audit procedure
Evaluate the base and aligned model on the same MMLU, GSM8K and HumanEval subsets plus a harmful-prompt refusal rate. Report delta and relative tax. A tolerance of 2% is a default for discussion; the squad should agree its own and justify it. Remember that SFT, DPO and steering each have a different tax profile; steering has none on the weights but can change outputs when attached.

## 5. Honest limits
- Written without PyTorch available in the build environment: the torch code was syntax-checked and its formulas checked against the numpy mirror, but the PyTorch tests and engine had not been run when this was packaged. Run `run_tests.py` first and fix anything it reports.
- Toy model, single seed; no real LLM, benchmark or GitHub repository is included.
