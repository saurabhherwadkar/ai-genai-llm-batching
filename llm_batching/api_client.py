# api_client.py
# Handles HTTP communication with the LLM API endpoint.
# Manages connection pooling, timeouts, and retry logic with
# exponential backoff for transient failures.

import asyncio
import logging

import httpx

from llm_batching.config import AppConfig

# Module-level logger for API client diagnostics
logger = logging.getLogger(__name__)


class ApiClient:
    """
    Asynchronous HTTP client for communicating with the LLM API.

    Manages connection lifecycle, applies authentication headers,
    and implements retry logic with exponential backoff for
    transient network and server errors.

    Attributes:
        config: Application configuration with API connection parameters.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the API client with connection configuration.

        Args:
            config: Application configuration containing API URL, key, and timeout settings.
        """
        # Store configuration for connection parameters
        self._config = config
        # Initialize the HTTP client reference (created on first use)
        self._client: httpx.AsyncClient | None = None
        # Log client initialization
        logger.info("ApiClient initialized for base_url=%s", config.api_base_url)

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Get or create the async HTTP client instance.

        Lazily creates the client on first access to support
        proper async context management.

        Returns:
            The configured httpx.AsyncClient instance.
        """
        # Create client if it doesn't exist yet
        if self._client is None:
            # Configure the client with base URL, auth headers, and timeout
            self._client = httpx.AsyncClient(
                base_url=self._config.api_base_url,
                headers=self._build_headers(),
                timeout=httpx.Timeout(self._config.request_timeout_seconds),
            )
            # Log client creation
            logger.debug("HTTP client created with timeout=%ss", self._config.request_timeout_seconds)
        return self._client

    def _build_headers(self) -> dict[str, str]:
        """
        Construct the HTTP headers for API authentication.

        Builds the Authorization header with the configured API key
        and sets the content type for JSON payloads.

        Returns:
            Dictionary of HTTP headers for the API requests.
        """
        # Build headers with Bearer token authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }
        return headers

    async def send_request(self, payload: dict) -> dict:
        """
        Send a single request to the LLM API with retry logic.

        Implements exponential backoff for transient failures (5xx errors,
        timeouts, and network errors). Returns the parsed JSON response
        on success or raises after exhausting all retries.

        Args:
            payload: The JSON request payload to send to the chat completions endpoint.

        Returns:
            Parsed JSON response dictionary from the API.

        Raises:
            ApiClientError: When all retry attempts are exhausted.
        """
        # Track the current retry attempt number
        attempt = 0
        # Store the last exception for error reporting
        last_error: Exception | None = None

        # Retry loop with configured maximum attempts
        while attempt <= self._config.max_retries:
            try:
                # Get the HTTP client instance
                client = await self._get_client()
                # Send POST request to the chat completions endpoint
                response = await client.post("/chat/completions", json=payload)
                # Raise exception for non-2xx HTTP status codes
                response.raise_for_status()
                # Parse and return the JSON response body
                json_response = response.json()
                # Log successful request
                logger.debug("API request successful on attempt %d", attempt + 1)
                return json_response

            except httpx.HTTPStatusError as http_error:
                # Log HTTP error with status code
                logger.warning(
                    "HTTP error on attempt %d: status=%d",
                    attempt + 1,
                    http_error.response.status_code,
                )
                # Store error for potential re-raise
                last_error = http_error
                # Only retry on server errors (5xx), not client errors (4xx)
                if http_error.response.status_code < 500:
                    break

            except (httpx.TimeoutException, httpx.ConnectError) as network_error:
                # Log network/timeout error
                logger.warning("Network error on attempt %d: %s", attempt + 1, network_error)
                # Store error for potential re-raise
                last_error = network_error

            # Increment the attempt counter
            attempt += 1

            # Apply exponential backoff delay before next retry
            if attempt <= self._config.max_retries:
                # Calculate delay with exponential backoff
                delay = self._config.retry_delay_seconds * (2 ** (attempt - 1))
                # Log the retry delay
                logger.info("Retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, self._config.max_retries + 1)
                # Wait before the next attempt
                await asyncio.sleep(delay)

        # All retries exhausted, raise the final error
        error_message = f"All {self._config.max_retries + 1} attempts failed: {last_error}"
        # Log the final failure
        logger.error(error_message)
        raise ApiClientError(error_message)

    async def close(self) -> None:
        """
        Close the HTTP client and release connection resources.

        Should be called when the client is no longer needed to
        properly release connection pool resources.
        """
        # Close the client if it was created
        if self._client is not None:
            # Close the underlying HTTP connection pool
            await self._client.aclose()
            # Clear the client reference
            self._client = None
            # Log client closure
            logger.info("API client connection closed")


class ApiClientError(Exception):
    """
    Custom exception for API client failures.

    Raised when all retry attempts are exhausted or when a
    non-retryable error occurs during API communication.
    """

    pass
