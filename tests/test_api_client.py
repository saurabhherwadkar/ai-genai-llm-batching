# test_api_client.py
# Unit tests for the ApiClient class.
# Uses respx to mock HTTP responses and verify retry logic,
# error handling, and request/response processing.

import httpx
import pytest
import respx

from llm_batching.api_client import ApiClient, ApiClientError


class TestApiClient:
    """Tests for ApiClient HTTP communication and retry logic."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_request_successful(self, test_config, sample_api_response):
        """Verify successful API call returns parsed JSON response."""
        # Mock the API endpoint to return a successful response
        respx.post(f"{test_config.api_base_url}/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_api_response)
        )
        # Create client and send request
        client = ApiClient(test_config)
        result = await client.send_request({"model": "test", "messages": []})
        # Assert the response matches the mocked data
        assert result == sample_api_response
        # Cleanup
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_request_retries_on_server_error(self, test_config, sample_api_response):
        """Verify client retries on 500 server errors and succeeds on retry."""
        # Mock first call to fail with 500, second to succeed
        route = respx.post(f"{test_config.api_base_url}/chat/completions")
        route.side_effect = [
            httpx.Response(500, json={"error": "Internal Server Error"}),
            httpx.Response(200, json=sample_api_response),
        ]
        # Create client and send request
        client = ApiClient(test_config)
        result = await client.send_request({"model": "test", "messages": []})
        # Assert success after retry
        assert result == sample_api_response
        # Assert two calls were made (initial + 1 retry)
        assert route.call_count == 2
        # Cleanup
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_request_fails_after_max_retries(self, test_config):
        """Verify ApiClientError is raised after all retries are exhausted."""
        # Mock all calls to return 500 errors
        respx.post(f"{test_config.api_base_url}/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "Server Error"})
        )
        # Create client and attempt request
        client = ApiClient(test_config)
        # Assert ApiClientError is raised after exhausting retries
        with pytest.raises(ApiClientError):
            await client.send_request({"model": "test", "messages": []})
        # Cleanup
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_request_no_retry_on_client_error(self, test_config):
        """Verify 4xx errors are not retried (only 5xx are retryable)."""
        # Mock the endpoint to return a 401 unauthorized error
        route = respx.post(f"{test_config.api_base_url}/chat/completions")
        route.mock(return_value=httpx.Response(401, json={"error": "Unauthorized"}))
        # Create client and attempt request
        client = ApiClient(test_config)
        # Assert error is raised without retrying
        with pytest.raises(ApiClientError):
            await client.send_request({"model": "test", "messages": []})
        # Assert only one call was made (no retries for 4xx)
        assert route.call_count == 1
        # Cleanup
        await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_send_request_retries_on_timeout(self, test_config, sample_api_response):
        """Verify client retries on timeout exceptions."""
        # Mock first call to timeout, second to succeed
        route = respx.post(f"{test_config.api_base_url}/chat/completions")
        route.side_effect = [
            httpx.ReadTimeout("Connection timed out"),
            httpx.Response(200, json=sample_api_response),
        ]
        # Create client and send request
        client = ApiClient(test_config)
        result = await client.send_request({"model": "test", "messages": []})
        # Assert success after retry
        assert result == sample_api_response
        # Cleanup
        await client.close()

    @pytest.mark.asyncio
    async def test_close_client(self, test_config):
        """Verify client close releases resources without error."""
        # Create client
        client = ApiClient(test_config)
        # Close without having made any requests (no client created yet)
        await client.close()
        # Assert client reference is cleared
        assert client._client is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_close_after_request(self, test_config, sample_api_response):
        """Verify client close works after making requests."""
        # Mock successful response
        respx.post(f"{test_config.api_base_url}/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_api_response)
        )
        # Create client and make a request to initialize the HTTP client
        client = ApiClient(test_config)
        await client.send_request({"model": "test", "messages": []})
        # Close the client
        await client.close()
        # Assert client reference is cleared
        assert client._client is None
