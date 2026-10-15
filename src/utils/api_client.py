"""OpenAI-compatible client for structured model responses."""
import os
import time
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


class LLMClient:
    """Call an OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: int = 120, api_key_env: str = "OPENAI_API_KEY"):

        self.base_url = base_url or "https://api.openai.com/v1"
        self.timeout = timeout
        self.api_key_env = api_key_env

        if api_key is None:
            api_key = os.getenv(api_key_env)
            if api_key is None:
                raise ValueError(f"API key not provided and {api_key_env} environment variable not set")

        self.api_key = api_key
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

    def generate_json(
        self,
        messages: list,
        model: str,
        logprobs: bool = True,
        top_logprobs: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Dict[str, Any]:

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    logprobs=logprobs,
                    top_logprobs=top_logprobs if logprobs else None,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                content = completion.choices[0].message.content

                return {
                    "response": completion,
                    "content": content,
                    "error": None
                }

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                if attempt == max_retries - 1:
                    return {
                        "response": None,
                        "content": None,
                        "error": error_msg
                    }
                time.sleep(retry_delay)

        return {
            "response": None,
            "content": None,
            "error": "Max retries exceeded"
        }

    def switch_base_url(self, new_base_url: str, new_api_key: Optional[str] = None):

        self.base_url = new_base_url
        if new_api_key:
            self.api_key = new_api_key
        else:

            self.api_key = os.getenv(self.api_key_env, self.api_key)

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout
        )

