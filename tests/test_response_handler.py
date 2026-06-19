# test_response_handler.py
# Unit tests for the ResponseHandler class.
# Verifies response parsing, error handling, and result aggregation
# for both successful and failed API responses.

from llm_batching.models import BatchResponse
from llm_batching.response_handler import ResponseHandler


class TestResponseHandler:
    """Tests for ResponseHandler parsing and aggregation logic."""

    def test_parse_successful_response(self, sample_api_response):
        """Verify successful API response is parsed correctly."""
        # Initialize response handler
        handler = ResponseHandler()
        # Parse the sample API response
        response = handler.parse_response("req-1", sample_api_response)
        # Assert parsed values match the API response
        assert response.request_id == "req-1"
        assert response.content == "Python is a high-level programming language."
        assert response.tokens_used == 18
        assert response.model == "test-model"
        assert response.success is True
        assert response.error_message == ""

    def test_parse_response_with_empty_choices(self):
        """Verify parsing handles response with empty choices list."""
        # Initialize response handler
        handler = ResponseHandler()
        # Create response with empty choices
        raw_response = {"choices": [], "usage": {"total_tokens": 0}, "model": "test"}
        # Parse the response
        response = handler.parse_response("req-2", raw_response)
        # Assert empty content is returned
        assert response.content == ""
        assert response.success is True

    def test_parse_response_with_missing_usage(self):
        """Verify parsing handles response missing usage statistics."""
        # Initialize response handler
        handler = ResponseHandler()
        # Create response without usage field
        raw_response = {
            "choices": [{"message": {"content": "Hello"}}],
            "model": "test",
        }
        # Parse the response
        response = handler.parse_response("req-3", raw_response)
        # Assert tokens default to zero when usage is missing
        assert response.tokens_used == 0
        assert response.content == "Hello"

    def test_parse_response_with_malformed_data(self):
        """Verify parsing handles completely malformed response gracefully."""
        # Initialize response handler
        handler = ResponseHandler()
        # Create completely invalid response structure
        raw_response = {"unexpected": "format"}
        # Parse the response - should not raise, should return error
        response = handler.parse_response("req-4", raw_response)
        # Assert response indicates success with empty content (empty choices)
        assert response.content == ""
        assert response.success is True

    def test_parse_response_with_none_choices(self):
        """Verify parsing handles None in choices field."""
        # Initialize response handler
        handler = ResponseHandler()
        # Create response with invalid choices structure
        raw_response = {"choices": [{"message": None}], "model": "test"}
        # Parse the response - should handle TypeError gracefully
        response = handler.parse_response("req-5", raw_response)
        # Assert error response is returned
        assert response.success is False
        assert response.error_message != ""

    def test_parse_error_response(self):
        """Verify error response creation with message."""
        # Initialize response handler
        handler = ResponseHandler()
        # Create an error response
        response = handler.parse_error_response("req-6", "Connection refused")
        # Assert error response fields
        assert response.request_id == "req-6"
        assert response.success is False
        assert response.error_message == "Connection refused"
        assert response.content == ""
        assert response.tokens_used == 0

    def test_aggregate_results_all_successful(self):
        """Verify aggregation with all successful responses."""
        # Initialize response handler
        handler = ResponseHandler()
        # Create list of successful responses
        responses = [
            BatchResponse(request_id="r1", content="A", tokens_used=10, success=True),
            BatchResponse(request_id="r2", content="B", tokens_used=15, success=True),
            BatchResponse(request_id="r3", content="C", tokens_used=20, success=True),
        ]
        # Aggregate the responses
        result = handler.aggregate_results(responses)
        # Assert aggregation statistics
        assert result.total_requests == 3
        assert result.successful_count == 3
        assert result.failed_count == 0
        assert result.total_tokens_used == 45
        assert len(result.responses) == 3

    def test_aggregate_results_mixed_success_failure(self):
        """Verify aggregation with mixed success and failure responses."""
        # Initialize response handler
        handler = ResponseHandler()
        # Create mixed responses
        responses = [
            BatchResponse(request_id="r1", tokens_used=10, success=True),
            BatchResponse(request_id="r2", success=False, error_message="Timeout"),
            BatchResponse(request_id="r3", tokens_used=20, success=True),
        ]
        # Aggregate the responses
        result = handler.aggregate_results(responses)
        # Assert mixed statistics
        assert result.total_requests == 3
        assert result.successful_count == 2
        assert result.failed_count == 1
        assert result.total_tokens_used == 30

    def test_aggregate_results_empty_list(self):
        """Verify aggregation handles empty response list."""
        # Initialize response handler
        handler = ResponseHandler()
        # Aggregate empty list
        result = handler.aggregate_results([])
        # Assert zero values for empty batch
        assert result.total_requests == 0
        assert result.successful_count == 0
        assert result.failed_count == 0
        assert result.total_tokens_used == 0
