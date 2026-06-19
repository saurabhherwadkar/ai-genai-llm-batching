# response_handler.py
# Responsible for parsing and validating LLM API responses.
# Transforms raw HTTP response data into structured BatchResponse
# objects, handling both successful and error response formats.

import logging

from llm_batching.models import BatchResponse, BatchResult

# Module-level logger for response handling diagnostics
logger = logging.getLogger(__name__)


class ResponseHandler:
    """
    Parses and validates raw LLM API responses into structured models.

    Handles extraction of generated content, token usage, and error
    information from the API response format.
    """

    def parse_response(self, request_id: str, raw_response: dict) -> BatchResponse:
        """
        Parse a raw API response dictionary into a BatchResponse model.

        Extracts the generated text content, token usage statistics, and
        model information from the OpenAI-compatible response format.

        Args:
            request_id: The identifier of the original request this response belongs to.
            raw_response: The raw JSON response dictionary from the API.

        Returns:
            A validated BatchResponse with extracted content and metadata.
        """
        try:
            # Extract the list of generated choices from the response
            choices = raw_response.get("choices", [])
            # Get the first choice's message content, defaulting to empty string
            content = choices[0]["message"]["content"] if choices else ""
            # Extract token usage statistics from the response
            usage = raw_response.get("usage", {})
            # Calculate total tokens from prompt and completion counts
            total_tokens = usage.get("total_tokens", 0)
            # Get the model identifier from the response
            model = raw_response.get("model", "")

            # Build the successful response object
            response = BatchResponse(
                request_id=request_id,
                content=content,
                tokens_used=total_tokens,
                model=model,
                success=True,
            )
            # Log successful parsing with token count
            logger.debug("Parsed response for request id=%s, tokens=%d", request_id, total_tokens)
            return response

        except (KeyError, IndexError, TypeError) as parse_error:
            # Log the parsing failure for debugging
            logger.error("Failed to parse response for request id=%s: %s", request_id, parse_error)
            # Return a failure response with the error details
            return self._create_error_response(request_id, str(parse_error))

    def parse_error_response(self, request_id: str, error_message: str) -> BatchResponse:
        """
        Create a BatchResponse representing a failed API call.

        Used when the HTTP request itself fails (timeout, network error,
        non-2xx status code) rather than a parsing failure.

        Args:
            request_id: The identifier of the failed request.
            error_message: Description of the error that occurred.

        Returns:
            A BatchResponse marked as failed with the error message.
        """
        # Log the error response creation
        logger.warning("Creating error response for request id=%s: %s", request_id, error_message)
        # Delegate to the internal error response builder
        return self._create_error_response(request_id, error_message)

    def aggregate_results(self, responses: list[BatchResponse]) -> BatchResult:
        """
        Aggregate a list of individual responses into a BatchResult summary.

        Calculates totals for successful/failed counts and token usage
        across all responses in the batch.

        Args:
            responses: List of BatchResponse objects to aggregate.

        Returns:
            A BatchResult with computed summary statistics.
        """
        # Count responses that completed successfully
        successful_count = sum(1 for response in responses if response.success)
        # Count responses that failed
        failed_count = sum(1 for response in responses if not response.success)
        # Sum total tokens used across all successful responses
        total_tokens = sum(response.tokens_used for response in responses)

        # Build the aggregated result object
        result = BatchResult(
            total_requests=len(responses),
            successful_count=successful_count,
            failed_count=failed_count,
            total_tokens_used=total_tokens,
            responses=responses,
        )
        # Log the aggregation summary
        logger.info(
            "Batch result: total=%d, success=%d, failed=%d, tokens=%d",
            result.total_requests,
            successful_count,
            failed_count,
            total_tokens,
        )
        return result

    def _create_error_response(self, request_id: str, error_message: str) -> BatchResponse:
        """
        Build a standardized error BatchResponse object.

        Internal helper that creates a consistently formatted error
        response used by both parse_response and parse_error_response.

        Args:
            request_id: The identifier of the failed request.
            error_message: Description of the error.

        Returns:
            A BatchResponse marked as failed with empty content.
        """
        # Construct the error response with failure flag and message
        return BatchResponse(
            request_id=request_id,
            content="",
            tokens_used=0,
            model="",
            success=False,
            error_message=error_message,
        )
