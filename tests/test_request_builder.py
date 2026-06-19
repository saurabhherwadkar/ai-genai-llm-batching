# test_request_builder.py
# Unit tests for the RequestBuilder class.
# Verifies request creation, batch building, and API payload
# construction with configured defaults and custom values.

from llm_batching.models import RequestStatus
from llm_batching.request_builder import RequestBuilder


class TestRequestBuilder:
    """Tests for RequestBuilder request creation and payload building."""

    def test_create_request_with_defaults(self, test_config):
        """Verify request creation applies configuration defaults."""
        # Initialize builder with test configuration
        builder = RequestBuilder(test_config)
        # Create a request with just a prompt
        request = builder.create_request("What is Python?")
        # Assert defaults from config are applied
        assert request.prompt == "What is Python?"
        assert request.model == "test-model"
        assert request.max_tokens == 100
        assert request.temperature == 0.5
        assert request.status == RequestStatus.PENDING
        # Assert a UUID was generated for request_id
        assert len(request.request_id) > 0

    def test_create_request_with_custom_id(self, test_config):
        """Verify custom request ID is preserved when provided."""
        # Initialize builder with test configuration
        builder = RequestBuilder(test_config)
        # Create a request with a custom ID
        request = builder.create_request("Test prompt", request_id="custom-id-123")
        # Assert the custom ID was used
        assert request.request_id == "custom-id-123"

    def test_create_batch_from_multiple_prompts(self, test_config, sample_prompts):
        """Verify batch creation produces correct number of requests."""
        # Initialize builder with test configuration
        builder = RequestBuilder(test_config)
        # Create a batch from sample prompts
        requests = builder.create_batch(sample_prompts)
        # Assert correct number of requests created
        assert len(requests) == len(sample_prompts)
        # Assert each request has a unique ID
        request_ids = [r.request_id for r in requests]
        assert len(set(request_ids)) == len(sample_prompts)

    def test_create_batch_preserves_prompt_order(self, test_config):
        """Verify batch creation maintains the original prompt order."""
        # Initialize builder with test configuration
        builder = RequestBuilder(test_config)
        # Create batch with ordered prompts
        prompts = ["First", "Second", "Third"]
        requests = builder.create_batch(prompts)
        # Assert order is preserved
        assert requests[0].prompt == "First"
        assert requests[1].prompt == "Second"
        assert requests[2].prompt == "Third"

    def test_create_batch_empty_list(self, test_config):
        """Verify batch creation handles empty input gracefully."""
        # Initialize builder with test configuration
        builder = RequestBuilder(test_config)
        # Create batch with empty list
        requests = builder.create_batch([])
        # Assert empty list is returned
        assert requests == []

    def test_build_api_payload_structure(self, test_config, sample_batch_request):
        """Verify API payload has correct OpenAI-compatible structure."""
        # Initialize builder with test configuration
        builder = RequestBuilder(test_config)
        # Build payload from sample request
        payload = builder.build_api_payload(sample_batch_request)
        # Assert payload structure matches OpenAI format
        assert payload["model"] == "test-model"
        assert payload["max_tokens"] == 100
        assert payload["temperature"] == 0.5
        # Assert messages array has correct format
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "What is Python?"

    def test_build_api_payload_excludes_request_metadata(self, test_config, sample_batch_request):
        """Verify API payload does not include internal fields like request_id."""
        # Initialize builder with test configuration
        builder = RequestBuilder(test_config)
        # Build payload from sample request
        payload = builder.build_api_payload(sample_batch_request)
        # Assert internal fields are not in the payload
        assert "request_id" not in payload
        assert "status" not in payload
