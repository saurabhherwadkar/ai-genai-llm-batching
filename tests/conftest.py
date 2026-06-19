# conftest.py
# Shared pytest fixtures for the LLM batching test suite.
# Provides reusable configuration, mock data, and test helpers
# used across multiple test modules.

import pytest

from llm_batching.config import AppConfig
from llm_batching.models import BatchRequest, BatchResponse


@pytest.fixture
def test_config() -> AppConfig:
    """
    Provide a test configuration with safe defaults for unit testing.

    Returns:
        AppConfig instance configured for fast, isolated testing.
    """
    # Create config with short timeouts and no retries for fast tests
    return AppConfig(
        api_base_url="https://api.test.example.com/v1",
        api_key="test-api-key-12345",
        batch_size=3,
        max_concurrent_requests=2,
        request_timeout_seconds=5.0,
        max_retries=1,
        retry_delay_seconds=0.1,
        default_model="test-model",
        default_max_tokens=100,
        default_temperature=0.5,
        log_level="DEBUG",
    )


@pytest.fixture
def sample_prompts() -> list[str]:
    """
    Provide a list of sample prompts for batch processing tests.

    Returns:
        List of test prompt strings.
    """
    # Return a variety of test prompts
    return [
        "What is Python?",
        "Explain REST APIs.",
        "What is Docker?",
        "Define microservices.",
        "What is CI/CD?",
    ]


@pytest.fixture
def sample_batch_request() -> BatchRequest:
    """
    Provide a single sample BatchRequest for unit tests.

    Returns:
        A fully populated BatchRequest instance.
    """
    # Create a sample request with known values for assertions
    return BatchRequest(
        request_id="test-req-001",
        prompt="What is Python?",
        model="test-model",
        max_tokens=100,
        temperature=0.5,
    )


@pytest.fixture
def sample_api_response() -> dict:
    """
    Provide a sample successful API response in OpenAI format.

    Returns:
        Dictionary mimicking the OpenAI chat completions response format.
    """
    # Construct a realistic API response structure
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Python is a high-level programming language.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        },
    }


@pytest.fixture
def sample_batch_response() -> BatchResponse:
    """
    Provide a sample successful BatchResponse for unit tests.

    Returns:
        A BatchResponse instance representing a successful API call.
    """
    # Create a sample successful response
    return BatchResponse(
        request_id="test-req-001",
        content="Python is a high-level programming language.",
        tokens_used=18,
        model="test-model",
        success=True,
    )


@pytest.fixture
def sample_error_response() -> BatchResponse:
    """
    Provide a sample failed BatchResponse for unit tests.

    Returns:
        A BatchResponse instance representing a failed API call.
    """
    # Create a sample error response
    return BatchResponse(
        request_id="test-req-002",
        content="",
        tokens_used=0,
        model="",
        success=False,
        error_message="Connection timeout after 5s",
    )
