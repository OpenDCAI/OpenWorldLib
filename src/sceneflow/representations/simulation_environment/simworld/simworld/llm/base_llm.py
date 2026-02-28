"""Base LLM class for handling interactions with language models."""

import inspect
import os
import time
from typing import Optional

import openai

from ..utils.logger import Logger

from .retry import retry_api_call


class LLMMetaclass(type):
    """Metaclass to automatically add retry decorators to public methods."""
    def __new__(cls, name, bases, attrs):
        """Create a new class."""
        # Process all attributes that are functions
        for attr_name, attr_value in attrs.items():
            # Only process public methods (not starting with _)
            if not attr_name.startswith('_') and inspect.isfunction(attr_value):
                # Apply retry decorator to the method
                attrs[attr_name] = retry_api_call()(attr_value)

        return super().__new__(cls, name, bases, attrs)


class BaseLLM(metaclass=LLMMetaclass):
    """Base class for interacting with language models through OpenAI-compatible APIs."""

    def __init__(
        self,
        model_name: str,
        url: Optional[str] = None,
        provider: Optional[str] = 'openai'
    ):
        """Initialize the LLM client. Default uses OpenAI's API.

        Args:
            model_name: Name of the model to use.
            url: Base URL for the API. If None, uses OpenAI's default URL.
            provider: Provider to use. Can be 'openai', 'openrouter', or 'local'.
                      Use 'local' for vLLM and other local OpenAI-compatible servers.

        Raises:
            ValueError: If no valid API key is provided or if the URL is invalid.
        """
        # Get API key from environment if not provided
        openai_api_key = os.getenv('OPENAI_API_KEY')
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

        self.provider = provider

        if provider == 'openai':
            if not openai_api_key:
                raise ValueError('No OpenAI API key provided. Please set OPENAI_API_KEY environment variable.')
            self.api_key = openai_api_key
        elif provider == 'openrouter':
            if not openrouter_api_key:
                raise ValueError('No OpenRouter API key provided. Please set OPENROUTER_API_KEY environment variable.')
            self.api_key = openrouter_api_key
        elif provider == 'local':
            # For local models (vLLM, etc.), API key is not required
            self.api_key = os.getenv('OPENAI_API_KEY', 'not-needed')
        else:
            raise ValueError(f'Not supported provider: {provider}')

        if url == 'None':
            url = None

        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=url,
            )
            # Validate the API key for cloud providers
            # Skip validation for local providers as they may not implement models.list()
            if provider != 'local':
                self.client.models.list()
        except Exception as e:
            raise ValueError(f'Failed to initialize OpenAI client: {str(e)}')

        self.model_name = model_name
        self.logger = Logger.get_logger('BaseLLM')
        self.logger.info(f'Initialized LLM client for model -- {model_name}, url -- {url or "default"}')

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.5,
        top_p: float = None,
        **kwargs,
    ) -> str | None:
        """Generate text using the language model.

        Args:
            system_prompt: System prompt to guide model behavior.
            user_prompt: User input prompt.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Top p sampling parameter.

        Returns:
            Generated text response or None if generation fails.
        """
        start_time = time.time()
        try:
            response = self._generate_text_with_retry(
                system_prompt,
                user_prompt,
                max_tokens,
                temperature,
                top_p,
                **kwargs,
            )
            return response, time.time() - start_time
        except Exception:
            return None, time.time() - start_time

    def _generate_text_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.5,
        top_p: float = None,
        **kwargs,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return response.choices[0].message.content
