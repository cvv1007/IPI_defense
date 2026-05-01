# Defense Against Indirect Prompt Injection
This work and the code base is based off of the paper InjecAgent.

Main code lives in `src/`.

## Preparing the code
Install dependencies with
```bash
pip install -r requirements.txt
```

For API backed models, set the matching API key in the environment.
For local Hugging Face Llama style models, use a model name that can be loaded locally through `transformers`.

## Prompted evaluation
A small scale prompted run can be launched with
```bash
export PYTHONPATH=.
python -m src.evaluate_prompted_agent \
  --model_type Llama \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --setting base \
  --prompt_type InjecAgent \
  --candidate c0 \
  --mini \
  --shuffle
```

Candidate meanings:
- `c0`: baseline prompt, raw tool outputs, no gate
- `c1`: prompt hierarchy plus untrusted wrapper
- `c2`: prompt hierarchy plus wrapper plus deterministic gate

## AI search over defenses
Searches over serialized candidates of the form `C = (P, W, G)` where `P` is the prompt configuration, `W` is the wrapper mode, and `G` is a deterministic gating policy.

Reduced scale search run is
```bash
export PYTHONPATH=.
python -m src.search_defense \
  --model_type Llama \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --setting base \
  --prompt_type InjecAgent \
  --dev_dh 10 \
  --dev_ds 10 \
  --dev_benign 10 \
  --test_dh 5 \
  --test_ds 5 \
  --test_benign 5 \
  --max_rounds 3 \
  --max_candidates 12
```

The search script writes a summary, candidate history, best dev candidate, and held out finalist results under `results/search_*`.
