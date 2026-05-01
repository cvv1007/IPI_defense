# Defense Against Indirect Prompt Injection
This work and the code base is based off of the work from the paper: InjecAgent  

Main code is in */src/* which completes the testing loop

## Preparing the code

To install required dependencies:
```bash
pip install -r requirements.txt
```
Note: The codebase requires working API keys for whichever model chosen to test on

## Small Demo and Reproduce Key Findings

Using for example the Llama-3.1-8B-Instruct model, key results can be reproduced on a small scale with the following
```python
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
Where --mini flag indicates small scale testing (fewer test cases)  

--candidate flag indicates which protocols to use, same as in design document  
c0: Basic setting  
c1: Prompt hierarchy + Wrapper  
c2: Prompt hierarchy + Wrapper + Gating policies  
