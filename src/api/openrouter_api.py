import os
import logging
from copy import deepcopy
from collections import defaultdict
from typing import Generator, List, Dict

from openai import OpenAI

from .llm_api import LLMAPI
from ..utils import log_exception

_logger = logging.getLogger(__name__)


class OpenRouterAPI(LLMAPI):
    supported_models = [
        "kimi-k2",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gpt-5.5",
        "claude-sonnet-4.6",
        "gemini-3.1-pro-preview",
        "deepseek-v4-pro",
        "qwen3.6-max-preview",
        "qwen3.6-plus",
        "glm-5.1",
        "gpt-5.4-mini",
        "gemini-3.1-flash-lite-preview",
        "deepseek-v4-flash",
    ]

    model_map = {
        "kimi-k2": "moonshotai/kimi-k2",
        "gemini-2.5-pro": "google/gemini-2.5-pro",
        "gemini-2.5-flash": "google/gemini-2.5-flash",
        "gpt-5.5": "openai/gpt-5.5",
        "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
        "gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
        "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
        "qwen3.6-max-preview": "qwen/qwen3.6-max-preview",
        "qwen3.6-plus": "qwen/qwen3.6-plus",
        "glm-5.1": "z-ai/glm-5.1",
        "gpt-5.4-mini": "openai/gpt-5.4-mini",
        "gemini-3.1-flash-lite-preview": "google/gemini-3.1-flash-lite-preview",
        "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    }

    def __init__(self, model='gpt-5.4-mini', max_tokens=1024, n=1, temperature=1.0, top_p=1.0):
        self.n = n
        self.top_p = top_p
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(self, messages: List | str, **kwargs) -> Generator[str, None, Dict]:
        """
        Keyword Args:
            model (str): The model name to use. Default is 'gpt-5.4-mini'.
            max_tokens (int): The maximum number of tokens to generate. Default is 1024.
            temperature (float): Sampling temperature. Default is 1.0.
            top_p (float): Nucleus sampling probability. Default is 1.0.
            n (int): Number of completions to generate. OpenRouter compatibility is
                implemented by making repeated single-completion requests.
        """
        yield from []
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        model = kwargs.get('model', self.model)
        openrouter_model = self.model_map.get(model, model)
        self._set_proxy_env()
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers=self._default_headers(),
        )
        payload = {
            "messages": messages,
            "model": openrouter_model,
            "n": 1,
            "stream": False,
            "top_p": kwargs.get('top_p', self.top_p),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
            "temperature": kwargs.get('temperature', self.temperature),
        }
        if model in ["gemini-2.5-pro", "gemini-3.1-pro-preview"]:
            payload["extra_body"] = {"reasoning": {"max_tokens": kwargs.get('reasoning_max_tokens', 128)}}

        usage = dict(token=defaultdict(float), price=defaultdict(float))
        responses = []
        for _ in range(kwargs.get('n', self.n)):
            try:
                completion = client.chat.completions.create(**payload)
            except Exception as e:
                _logger.error(f"Error requesting {type(self).__name__}({model}) since: {log_exception(e)}")
                continue

            responses.append(completion.to_dict())
            self._accumulate_usage(usage, completion)
            text = completion.choices[0].message.content or ""
            yield text

        return {
            "usage": usage,
            "messages": messages,
            "contents": [response['choices'][0]['message'].get('content') or "" for response in responses],
            "responses": responses,
        }

    def _set_proxy_env(self) -> None:
        http_proxy = os.environ.get("MY_HTTP_PROXY")
        https_proxy = os.environ.get("MY_HTTPS_PROXY")
        if http_proxy:
            os.environ["HTTP_PROXY"] = http_proxy
            os.environ["http_proxy"] = http_proxy
        if https_proxy:
            os.environ["HTTPS_PROXY"] = https_proxy
            os.environ["https_proxy"] = https_proxy

    def _default_headers(self) -> Dict[str, str]:
        headers = {}
        if referer := os.environ.get("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = referer
        if title := os.environ.get("OPENROUTER_APP_TITLE"):
            headers["X-OpenRouter-Title"] = title
        return headers

    def _accumulate_usage(self, usage: Dict, completion) -> None:
        if completion.usage is None:
            return
        prompt_tokens = getattr(completion.usage, "prompt_tokens", 0) or 0
        total_tokens = getattr(completion.usage, "total_tokens", 0) or 0
        completion_tokens = getattr(completion.usage, "completion_tokens", 0) or 0
        prompt_details = getattr(completion.usage, "prompt_tokens_details", None)
        cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
        completion_details = getattr(completion.usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0
        usage['token']['cached'] += cached_tokens
        usage['token']['prompt'] += prompt_tokens - cached_tokens
        usage['token']['reason'] += reasoning_tokens
        usage['token']['answer'] += completion_tokens - reasoning_tokens
        if (other := total_tokens - prompt_tokens - completion_tokens) > 0:
            usage['token']['other'] += other
        if (cost := getattr(completion.usage, "cost", None)) is not None:
            usage['price']['total'] += cost
