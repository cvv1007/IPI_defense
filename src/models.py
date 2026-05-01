import os
import time


class BaseModel:
    def __init__(self):
        self.model = None

    def prepare_input(self, sys_prompt, user_prompt_filled):
        raise NotImplementedError

    def call_model(self, model_input):
        raise NotImplementedError


class ClaudeModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
        self.anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.params = params
        self.human_prompt = HUMAN_PROMPT
        self.ai_prompt = AI_PROMPT

    def prepare_input(self, sys_prompt, user_prompt_filled):
        return f"{self.human_prompt} {sys_prompt} {user_prompt_filled}{self.ai_prompt}"

    def call_model(self, model_input):
        completion = self.anthropic.completions.create(
            model=self.params['model_name'],
            max_tokens_to_sample=self.params.get('max_new_tokens', 256),
            prompt=model_input,
            temperature=0
        )
        return completion.completion


class GPTModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            organization=os.environ.get("OPENAI_ORGANIZATION")
        )
        self.params = params

    def prepare_input(self, sys_prompt, user_prompt_filled):
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_filled}
        ]

    def call_model(self, model_input):
        completion = self.client.chat.completions.create(
            model=self.params['model_name'],
            messages=model_input,
            temperature=0,
            max_tokens=self.params.get('max_new_tokens', 256)
        )
        return completion.choices[0].message.content


class TogetherAIModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        import together
        from src.prompts.prompt_template import PROMPT_TEMPLATE
        self.together = together
        self.params = params
        self.model = self.params['model_name']
        self.prompt = PROMPT_TEMPLATE[self.model]

    def prepare_input(self, sys_prompt, user_prompt_filled):
        return self.prompt.format(sys_prompt=sys_prompt, user_prompt=user_prompt_filled)

    def call_model(self, model_input, retries=3, delay=2):
        attempt = 0
        max_tokens = self.params.get('max_new_tokens', 256)
        while attempt < retries:
            try:
                completion = self.together.Complete.create(
                    model=self.model,
                    prompt=model_input,
                    max_tokens=max_tokens,
                    temperature=0
                )
                return completion['choices'][0]['text']
            except Exception as e:
                attempt += 1
                max_tokens = max(64, max_tokens // 2)
                print(f"Attempt {attempt}: An error occurred - {e}")
                if attempt < retries:
                    time.sleep(delay)
                else:
                    return ""


class LlamaModel(BaseModel):
    def __init__(self, params):
        super().__init__()
        import torch
        import transformers

        self.params = params
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(params['model_name'])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=params['model_name'],
            tokenizer=self.tokenizer,
            torch_dtype=torch_dtype,
            device_map="auto",
            do_sample=False,
            temperature=0,
            max_new_tokens=params.get('max_new_tokens', 128),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self._get_terminators(),
        )

    def _get_terminators(self):
        terminators = [self.tokenizer.eos_token_id]
        for token in ["<|eot_id|>", "<|end_of_text|>"]:
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None
            if token_id is not None and token_id != self.tokenizer.unk_token_id and token_id not in terminators:
                terminators.append(token_id)
        if len(terminators) == 1:
            return terminators[0]
        return terminators

    def prepare_input(self, sys_prompt, user_prompt_filled):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_filled}
        ]
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_prompt_filled} [/INST]"

    def call_model(self, model_input):
        output = self.pipeline(model_input, return_full_text=False)
        if not output:
            return ""
        return output[0].get("generated_text", "").strip()


MODELS = {
    "Claude": ClaudeModel,
    "GPT": GPTModel,
    "Llama": LlamaModel,
    "TogetherAI": TogetherAIModel
}