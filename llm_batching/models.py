# models.py
# Defines data models for LLM batch requests and responses.
# Uses Pydantic for validation and serialization of structured data
# flowing through the batching pipeline.

from enum import Enum

from pydantic import BaseModel, Field


class RequestStatus(str, Enum):
    """Represents the processing status of a single batch request item."""

    # Request is waiting to be processed
    PENDING = "pending"
    # Request is currently being sent to the LLM API
    IN_PROGRESS = "in_progress"
    # Request completed successfully with a response
    COMPLETED = "completed"
    # Request failed after all retry attempts
    FAILED = "failed"


class BatchRequest(BaseModel):
    """
    Represents a single prompt request within a batch.

    Attributes:
        request_id: Unique identifier for tracking this request through the pipeline.
        prompt: The text prompt to send to the LLM API.
        model: The LLM model identifier to use for this request.
        max_tokens: Maximum number of tokens to generate in the response.
        temperature: Sampling temperature controlling response randomness (0.0 to 1.0).
        status: Current processing status of this request.
    """

    # Unique identifier for tracking this request through the pipeline
    request_id: str = Field(description="Unique identifier for this request")
    # The text prompt to send to the LLM API
    prompt: str = Field(description="The text prompt to send to the LLM")
    # The LLM model identifier to use
    model: str = Field(default="gpt-3.5-turbo", description="Model identifier")
    # Maximum tokens to generate in the response
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Max tokens to generate")
    # Sampling temperature controlling randomness
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Sampling temperature")
    # Current processing status
    status: RequestStatus = Field(default=RequestStatus.PENDING, description="Processing status")


class BatchResponse(BaseModel):
    """
    Represents the response for a single request in the batch.

    Attributes:
        request_id: The matching request identifier this response belongs to.
        content: The generated text content from the LLM.
        tokens_used: Total number of tokens consumed (prompt + completion).
        model: The model that generated this response.
        success: Whether the request completed successfully.
        error_message: Description of the error if the request failed.
    """

    # The matching request identifier this response belongs to
    request_id: str = Field(description="Matching request ID")
    # The generated text content from the LLM
    content: str = Field(default="", description="Generated text content")
    # Total tokens consumed (prompt + completion)
    tokens_used: int = Field(default=0, ge=0, description="Total tokens consumed")
    # The model that generated this response
    model: str = Field(default="", description="Model that generated the response")
    # Whether the request completed successfully
    success: bool = Field(default=True, description="Whether request succeeded")
    # Error message if the request failed
    error_message: str = Field(default="", description="Error description if failed")


class BatchResult(BaseModel):
    """
    Aggregated result of processing an entire batch of requests.

    Attributes:
        total_requests: Total number of requests submitted in the batch.
        successful_count: Number of requests that completed successfully.
        failed_count: Number of requests that failed after retries.
        total_tokens_used: Sum of all tokens consumed across all responses.
        responses: List of individual response objects for each request.
    """

    # Total number of requests submitted in the batch
    total_requests: int = Field(description="Total requests in the batch")
    # Number of requests that completed successfully
    successful_count: int = Field(default=0, description="Successful request count")
    # Number of requests that failed after retries
    failed_count: int = Field(default=0, description="Failed request count")
    # Sum of all tokens consumed across all responses
    total_tokens_used: int = Field(default=0, description="Total tokens consumed")
    # List of individual response objects
    responses: list[BatchResponse] = Field(default_factory=list, description="Individual responses")
