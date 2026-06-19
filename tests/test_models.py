# test_models.py
# Unit tests for the Pydantic data models used in the batching pipeline.
# Covers validation, defaults, and edge cases for BatchRequest,
# BatchResponse, and BatchResult models.

import pytest
from pydantic import ValidationError

from llm_batching.models import BatchRequest, BatchResponse, BatchResult, RequestStatus


class TestBatchRequest:
    """Tests for the BatchRequest model validation and defaults."""

    def test_create_request_with_required_fields(self):
        """Verify a request can be created with only required fields."""
        # Create request with minimum required fields
        request = BatchRequest(request_id="req-1", prompt="Hello")
        # Assert defaults are applied correctly
        assert request.request_id == "req-1"
        assert request.prompt == "Hello"
        assert request.model == "gpt-3.5-turbo"
        assert request.max_tokens == 256
        assert request.temperature == 0.7
        assert request.status == RequestStatus.PENDING

    def test_create_request_with_all_fields(self):
        """Verify a request can be created with all fields specified."""
        # Create request with all fields explicitly set
        request = BatchRequest(
            request_id="req-2",
            prompt="Test prompt",
            model="gpt-4",
            max_tokens=512,
            temperature=0.3,
            status=RequestStatus.IN_PROGRESS,
        )
        # Assert all values are stored correctly
        assert request.model == "gpt-4"
        assert request.max_tokens == 512
        assert request.temperature == 0.3
        assert request.status == RequestStatus.IN_PROGRESS

    def test_request_rejects_invalid_max_tokens(self):
        """Verify validation rejects max_tokens outside allowed range."""
        # Attempt to create request with max_tokens exceeding limit
        with pytest.raises(ValidationError):
            BatchRequest(request_id="req-3", prompt="Test", max_tokens=5000)

    def test_request_rejects_invalid_temperature(self):
        """Verify validation rejects temperature outside 0.0-1.0 range."""
        # Attempt to create request with temperature above maximum
        with pytest.raises(ValidationError):
            BatchRequest(request_id="req-4", prompt="Test", temperature=1.5)

    def test_request_rejects_negative_temperature(self):
        """Verify validation rejects negative temperature values."""
        # Attempt to create request with negative temperature
        with pytest.raises(ValidationError):
            BatchRequest(request_id="req-5", prompt="Test", temperature=-0.1)


class TestBatchResponse:
    """Tests for the BatchResponse model validation and defaults."""

    def test_create_successful_response(self):
        """Verify a successful response stores all fields correctly."""
        # Create a successful response with all fields
        response = BatchResponse(
            request_id="req-1",
            content="Generated text",
            tokens_used=25,
            model="gpt-4",
            success=True,
        )
        # Assert all fields are stored correctly
        assert response.request_id == "req-1"
        assert response.content == "Generated text"
        assert response.tokens_used == 25
        assert response.success is True
        assert response.error_message == ""

    def test_create_error_response(self):
        """Verify an error response stores failure details correctly."""
        # Create an error response with failure information
        response = BatchResponse(
            request_id="req-2",
            success=False,
            error_message="API timeout",
        )
        # Assert error fields are populated
        assert response.success is False
        assert response.error_message == "API timeout"
        assert response.content == ""
        assert response.tokens_used == 0

    def test_response_defaults(self):
        """Verify default values are applied for optional fields."""
        # Create response with only request_id
        response = BatchResponse(request_id="req-3")
        # Assert defaults
        assert response.content == ""
        assert response.tokens_used == 0
        assert response.model == ""
        assert response.success is True
        assert response.error_message == ""


class TestBatchResult:
    """Tests for the BatchResult aggregation model."""

    def test_create_result_with_responses(self):
        """Verify batch result aggregates response data correctly."""
        # Create a result with mixed success/failure responses
        responses = [
            BatchResponse(request_id="req-1", content="Text", tokens_used=10, success=True),
            BatchResponse(request_id="req-2", success=False, error_message="Error"),
        ]
        result = BatchResult(
            total_requests=2,
            successful_count=1,
            failed_count=1,
            total_tokens_used=10,
            responses=responses,
        )
        # Assert aggregation values
        assert result.total_requests == 2
        assert result.successful_count == 1
        assert result.failed_count == 1
        assert result.total_tokens_used == 10
        assert len(result.responses) == 2

    def test_empty_result(self):
        """Verify batch result handles zero requests correctly."""
        # Create an empty result
        result = BatchResult(total_requests=0)
        # Assert default values for empty batch
        assert result.successful_count == 0
        assert result.failed_count == 0
        assert result.total_tokens_used == 0
        assert result.responses == []


class TestRequestStatus:
    """Tests for the RequestStatus enumeration values."""

    def test_status_values(self):
        """Verify all expected status values exist."""
        # Check all enum values are accessible
        assert RequestStatus.PENDING == "pending"
        assert RequestStatus.IN_PROGRESS == "in_progress"
        assert RequestStatus.COMPLETED == "completed"
        assert RequestStatus.FAILED == "failed"
