# AURA_squad_submission4: Mission 4, Subsystem 04 (Plasma Ignition & Ethical Steering Core)

Sector SEC-04-ORI (Betelgeuse). Goal: raise consciousness 7.1% to 28.4% by implementing DPO, implicit-reward evaluation, SimPO and activation steering in PyTorch, then audit the alignment tax.

| Folder | Role | Objective | Main code |
|---|---|---|---|
| `01_Data_Specialist` | Data Specialist | [3] length-bias suite + SimPO | `simpo_loss.py`, `length_bias_diagnostics.py` |
| `02_Protocol_Specialist` | Protocol Officer | [2] pair comparison, implicit reward | `reward_engine.py` |
| `03_Security_Sentinel` | Security Sentinel | [4] contrast vectors, steering | `activation_steering.py` |
| `04_Systems_Architect` | Systems Architect | [1] DPO loss, integration [5] | `dpo_loss.py`, `ps04_engine.py`, `ps04_solution.ipynb` |

## Quick start
```
cd 04_Systems_Architect
python verify_math_numpy.py        # works without PyTorch
pip install torch numpy
python run_tests.py                # all role tests
python ps04_engine.py --demo all   # toy-model demos, writes outputs/
```
Each role folder has its own brief (`.md`). Not included: a GitHub URL (create it from your own account), real benchmark scores (template only), real LLM runs.
"# SQUAD_RUMELHART_MISSIO-4" 
