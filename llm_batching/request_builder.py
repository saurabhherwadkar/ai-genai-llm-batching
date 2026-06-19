# request_builder.py
# Responsible for constructing and validating LLM API request payloads.
# Transforms BatchRequest models into the HTTP request format expected
# by the LLM API, handling prompt formatting and parameter mapping.

import logging
import uuid

from llm_batching.config import AppConfig
from llm_batching.models import BatchRequest

# Module-level logger for request building diagnostics
logger = logging.getLogger(__name__)


class RequestBuilder:
    """
    Builds and validates LLM API request payloads from BatchRequest models.

    Handles the transformation of application-level request objects into
    the specific format required by the target LLM API endpoint.

    Attributes:
        config: Application configuration containing default model parameters.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the RequestBuilder with application configuration.

        Args:
            config: Application configuration with default model parameters.
        """
        # Store configuration for default parameter values
        self._config = config
        # Log initialization of the request builder
        logger.info("RequestBuilder initialized with model=%s", config.default_model)

    def create_request(self, prompt: str, request_id: str | None = None) -> BatchRequest:
        """
        Create a BatchRequest from a prompt string using configured defaults.

        Generates a unique request ID if not provided, and applies default
        model parameters from the application configuration.

        Args:
            prompt: The text prompt to send to the LLM API.
            request_id: Optional custom identifier; auto-generated if not provided.

        Returns:
            A validated BatchRequest instance ready for batching.
        """
        # Generate a unique request ID if none was provided
        generated_id = request_id or str(uuid.uuid4())
        # Create the batch request with configured defaults
        request = BatchRequest(
            request_id=generated_id,
            prompt=prompt,
            model=self._config.default_model,
            max_tokens=self._config.default_max_tokens,
            temperature=self._config.default_temperature,
        )
        # Log the created request for debugging
        logger.debug("Created request id=%s, prompt_length=%d", generated_id, len(prompt))
        return request

    def create_batch(self, prompts: list[str]) -> list[BatchRequest]:
        """
        Create a list of BatchRequest objects from multiple prompts.

        Each prompt receives a unique request ID and configured defaults.

        Args:
            prompts: List of text prompts to convert into batch requests.

        Returns:
            List of validated BatchRequest instances.
        """
        # Build a request for each prompt in the input list
        requests = [self.create_request(prompt) for prompt in prompts]
        # Log the batch creation summary
        logger.info("Created batch of %d requests", len(requests))
        return requests

    def build_api_payload(self, request: BatchRequest) -> dict:
        """
        Transform a BatchRequest into the API-specific HTTP request payload.

        Formats the request into the JSON structure expected by the
        OpenAI-compatible chat completions endpoint.

        Args:
            request: The BatchRequest to transform into API format.

        Returns:
            Dictionary representing the JSON payload for the API call.
        """
        # Construct the API payload in OpenAI chat completions format
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        # Log payload construction for debugging
        logger.debug("Built API payload for request id=%s", request.request_id)
        return payload
