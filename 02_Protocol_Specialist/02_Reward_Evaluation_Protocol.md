# 02 Reward Evaluation Protocol

**Station role:** Protocol Officer
**Mission 4 objective [2]:** construct a pair-comparison evaluation engine and derive implicit reward metrics across prompt-completion pairs.

## 1. Implicit reward
DPO needs no separate reward model. The policy itself defines one:

r_hat(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) ) + beta * log Z(x)

The Z(x) term depends only on the prompt, so it cancels when two answers to the same prompt are compared. `evaluate_implicit_reward` therefore returns beta * (log pi_theta - log pi_ref) and the docstring warns that single reward values are only meaningful within a prompt.

## 2. Evaluation protocol
1. For each pair, compute the summed log-probability of the chosen and rejected answers under the policy and under the frozen reference.
2. Convert to implicit rewards with `evaluate_implicit_reward`.
3. Margin = r(chosen) - r(rejected). Bradley-Terry probability P(chosen preferred) = sigmoid(margin).
4. Report reward accuracy (share of pairs with margin > 0), mean margin, mean probability and the number of exact ties.
5. A pair counts as correct only when margin is strictly greater than 0; ties count as wrong.

## 3. Files
| File | Purpose |
|---|---|
| `reward_engine.py` | `evaluate_implicit_reward`, `PairComparisonEngine.compare`, markdown table output. Validates tensor shapes and beta > 0. |
| `tests/test_reward_engine.py` | 5 PyTorch tests: formula, accuracy and probability, ties, shape error, bad beta (skipped if torch is missing). |

## 4. Reading the numbers
Before training, policy equals reference, so every margin is 0 and accuracy is 0 by the strict rule. After DPO training, accuracy should rise above 0.5 on held-out pairs. A rising margin on training pairs with flat held-out accuracy signals over-fitting. A healthy accuracy together with a high length-bias score (Data Specialist) means the reward is being gamed by length.

## 5. Honest limits
- Accuracy here is accuracy of the model's own implicit reward, not agreement with human judges.
- Pair data in the demo is synthetic.
