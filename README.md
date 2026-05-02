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
The repository now includes an AI driven search loop that follows the design document search structure. It searches over serialized candidates of the form `C = (P, W, G)` where `P` is the prompt configuration, `W` is the wrapper mode, and `G` is a deterministic gating policy.

The search uses a concrete benign success threshold of `BSR >= 60.0` by default through `--bmin 60.0`. It also prunes candidates with `Invalid Rate > 25.0` by default through `--imax 25.0`. These thresholds are written into `search_summary.json` for reproducibility.

A typical reduced scale search run is
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

If you also want an LLM based proposer instead of pure heuristic edits, add
```bash
  --search_model_type GPT \
  --search_model_name gpt-4o-mini
```

The search script writes a summary, candidate history, proposer inputs, best dev candidate, and held out finalist results under `results/search_*`. In particular, `proposer_inputs.jsonl` saves the exact system prompt, user prompt, parent candidate, parent metrics, failure traces, prepared model input, and raw proposer response for each search round so you can inspect what the search LLM actually received.

## Limitations
These defenses are prompt and policy layer defenses only. They can still be bypassed by rephrased or adaptive attacks because the base model was not trained to robustly separate trusted instructions from untrusted tool content. The deterministic gates reduce some risky actions, but they do not solve the underlying representation problem. Results should therefore be interpreted as deployable mitigations for InjecAgent style evaluation rather than a complete defense against indirect prompt injection.
