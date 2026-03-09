import pprint

from openai import OpenAI

from modules import shared
from modules.logging_colors import logger


class OpenAIAPIModel:
    """
    Model class that delegates inference to an external OpenAI-compatible API.
    Uses the official openai Python library with a custom base_url.
    Follows the same interface as LlamaServer for use with generate_reply_custom.
    """

    def __init__(self, api_url, api_key, model_name):
        self.api_url = api_url.rstrip('/')
        self.remote_model_name = model_name
        self.last_prompt_token_count = 0
        self.use_completions = False

        # The openai library expects the base_url to include /v1.
        # If the user already provided it (e.g., https://openrouter.ai/api/v1),
        # don't append it again.
        base_url = self.api_url
        if not base_url.endswith('/v1'):
            base_url = f"{base_url}/v1"

        # The openai library requires a non-empty api_key string
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or 'no-key',
        )

        # Verify connectivity
        self._verify_connection()

    def _verify_connection(self):
        """Check that the API endpoint is reachable."""
        try:
            self.client.models.list()
            logger.info(f"Connected to OpenAI-compatible API at {self.api_url}")
        except Exception as e:
            error_str = str(e)
            if '401' in error_str:
                raise RuntimeError(f"Authentication failed for {self.api_url}. Check your API key.")
            else:
                logger.warning(f"Could not verify API connection: {e}. Generation will be attempted anyway.")

    def encode(self, text, **kwargs):
        """Approximate token count. Uses tiktoken if available, else character-based estimation."""
        text = str(text)
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return enc.encode(text)
        except (ImportError, Exception):
            # Approximate: ~4 characters per token
            estimated_count = max(1, len(text) // 4)
            return list(range(estimated_count))

    def decode(self, token_ids, **kwargs):
        """Not meaningful for remote APIs, but needed for interface compatibility."""
        return ''

    def _build_params(self, prompt, state):
        """Convert UI state to OpenAI API request parameters."""
        max_tokens = state.get('max_new_tokens', 512)
        if state.get('auto_max_new_tokens'):
            max_tokens = max(1, state.get('truncation_length', 8192) - self.last_prompt_token_count)

        params = {
            'model': self.remote_model_name,
            'max_tokens': max_tokens,
            'stream': True,
        }

        # Map sampler parameters (only include non-default values)
        if state.get('temperature', 1) != 1:
            params['temperature'] = state['temperature']

        if state.get('top_p', 1) != 1:
            params['top_p'] = state['top_p']

        if state.get('presence_penalty', 0) != 0:
            params['presence_penalty'] = state['presence_penalty']

        if state.get('frequency_penalty', 0) != 0:
            params['frequency_penalty'] = state['frequency_penalty']

        if state.get('seed', -1) != -1:
            params['seed'] = state['seed']

        return params

    def generate_with_streaming(self, prompt, state):
        """Stream generation from the OpenAI-compatible API."""
        self.last_prompt_token_count = len(self.encode(prompt))
        params = self._build_params(prompt, state)

        if shared.args.verbose:
            logger.info("GENERATE_PARAMS=")
            pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(
                {k: v for k, v in params.items() if k not in ('prompt', 'messages')}
            )
            print()

        full_text = ""
        stop_event = state.get('stop_event')

        try:
            if self.use_completions:
                params['prompt'] = prompt
                stream = self.client.completions.create(**params)
                for chunk in stream:
                    if shared.stop_everything or (stop_event and stop_event.is_set()):
                        stream.close()
                        break

                    if chunk.choices and chunk.choices[0].text:
                        full_text += chunk.choices[0].text
                        yield full_text
            else:
                params['messages'] = [{'role': 'user', 'content': prompt}]
                stream = self.client.chat.completions.create(**params)
                for chunk in stream:
                    if shared.stop_everything or (stop_event and stop_event.is_set()):
                        stream.close()
                        break

                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        full_text += chunk.choices[0].delta.content
                        yield full_text

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return

    def generate(self, prompt, state):
        """Non-streaming generation."""
        output = ""
        for output in self.generate_with_streaming(prompt, state):
            pass
        return output

    def unload(self):
        """Clean up the client."""
        if self.client:
            self.client.close()
