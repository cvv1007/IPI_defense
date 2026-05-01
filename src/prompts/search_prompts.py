SEARCH_SYSTEM_PROMPT = """
You are assisting with a structured search over deployable defenses for indirect prompt injection in a tool using agent.
You must edit only the serialized candidate configuration and keep gates deterministic.
Never rewrite the full base prompt. Only return JSON.
"""


SEARCH_USER_PROMPT = """
You are given a parent candidate, its dev metrics, and a few failure traces.
Propose exactly one child candidate with at most {max_atomic_edits} atomic edits.

Allowed atomic edits:
1. Change prompt.phier to one of {prompt_hier_options}
2. Change prompt.pform to one of {prompt_form_options}
3. Change prompt.pstr to one of {prompt_strength_options}
4. Change wrapper_mode to one of {wrapper_options}
5. Change gate.scope to one of {gate_scope_options}
6. Change gate.action to one of {gate_action_options}
7. Change one of gate.action_threshold, gate.intent_threshold, gate.sensitivity_threshold by plus or minus 1 within [1, 5]
8. Change gate.mode between none and custom

Keep the candidate space serialized and compact.
Prefer edits that reduce attack success while preserving benign success and low invalid rate.
Do not remove deterministic gating when the main failures are harmful tool calls.

Return JSON with this schema and no extra text:
{{
  "rationale": "brief string",
  "edits": ["edit 1", "edit 2"],
  "candidate": {{
    "prompt": {{"phier": "...", "pform": "...", "pstr": "..."}},
    "wrapper_mode": "...",
    "gate": {{
      "mode": "none" or "custom",
      "scope": "...",
      "action": "...",
      "action_threshold": 1,
      "intent_threshold": 1,
      "sensitivity_threshold": 1
    }}
  }}
}}

Parent candidate:
{parent_candidate}

Parent metrics:
{parent_metrics}

Representative failure traces:
{failure_traces}
"""
