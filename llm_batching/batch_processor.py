# batch_processor.py
# Orchestrates the end-to-end LLM batch processing pipeline.
# Collects prompts, splits them into sized batches, processes each
# batch concurrently with rate limiting, and aggregates results.

import asyncio
import logging

from llm_batching.api_client import ApiClient, ApiClientError
from llm_batching.config import AppConfig
from llm_batching.models import BatchRequest, BatchResult, RequestStatus
from llm_batching.request_builder import RequestBuilder
from llm_batching.response_handler import ResponseHandler

# Module-level logger for batch processing diagnostics
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Orchestrates the complete LLM batch processing pipeline.

    Manages the lifecycle of batch requests: splitting prompts into
    sized batches, sending them concurrently with rate limiting via
    a semaphore, and aggregating all responses into a unified result.

    Attributes:
        config: Application configuration with batching parameters.
        request_builder: Constructs API request payloads.
        response_handler: Parses API responses into structured models.
        api_client: Handles HTTP communication with the LLM API.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize the BatchProcessor with all required dependencies.

        Creates the request builder, response handler, and API client
        using the provided configuration.

        Args:
            config: Application configuration with batching and API parameters.
        """
        # Store configuration for batch size and concurrency settings
        self._config = config
        # Initialize the request builder for creating API payloads
        self._request_builder = RequestBuilder(config)
        # Initialize the response handler for parsing API responses
        self._response_handler = ResponseHandler()
        # Initialize the API client for HTTP communication
        self._api_client = ApiClient(config)
        # Create semaphore to limit concurrent API requests
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        # Log processor initialization
        logger.info(
            "BatchProcessor initialized: batch_size=%d, max_concurrent=%d",
            config.batch_size,
            config.max_concurrent_requests,
        )

    async def process_prompts(self, prompts: list[str]) -> BatchResult:
        """
        Process a list of prompts through the complete batching pipeline.

        Splits prompts into batches, processes each batch concurrently,
        and returns an aggregated result with all responses.

        Args:
            prompts: List of text prompts to send to the LLM API.

        Returns:
            BatchResult containing all responses and summary statistics.
        """
        # Log the start of batch processing
        logger.info("Starting batch processing for %d prompts", len(prompts))

        # Create BatchRequest objects from the raw prompts
        requests = self._request_builder.create_batch(prompts)

        # Split requests into sized batches based on configuration
        batches = self._split_into_batches(requests)
        # Log the batch split result
        logger.info("Split into %d batches of max size %d", len(batches), self._config.batch_size)

        # Process all batches and collect responses
        all_responses = []
        # Iterate through each batch for processing
        for batch_index, batch in enumerate(batches):
            # Log the start of each batch
            logger.info("Processing batch %d/%d (%d requests)", batch_index + 1, len(batches), len(batch))
            # Process the current batch concurrently
            batch_responses = await self._process_batch(batch)
            # Add batch responses to the overall collection
            all_responses.extend(batch_responses)

        # Aggregate all responses into a unified result
        result = self._response_handler.aggregate_results(all_responses)
        # Log completion summary
        logger.info("Batch processing complete: %d/%d successful", result.successful_count, result.total_requests)

        return result

    def _split_into_batches(self, requests: list[BatchRequest]) -> list[list[BatchRequest]]:
        """
        Split a list of requests into sized batches.

        Divides the request list into chunks of the configured batch size
        for controlled processing.

        Args:
            requests: Full list of BatchRequest objects to split.

        Returns:
            List of batch lists, each containing at most batch_size requests.
        """
        # Get the configured batch size
        batch_size = self._config.batch_size
        # Split requests into chunks using list slicing
        batches = [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]
        return batches

    async def _process_batch(self, batch: list[BatchRequest]) -> list:
        """
        Process a single batch of requests concurrently with rate limiting.

        Uses asyncio.gather to process all requests in the batch simultaneously,
        with a semaphore limiting the maximum concurrent API calls.

        Args:
            batch: List of BatchRequest objects in this batch.

        Returns:
            List of BatchResponse objects for each request in the batch.
        """
        # Create async tasks for each request in the batch
        tasks = [self._process_single_request(request) for request in batch]
        # Execute all tasks concurrently and wait for all to complete
        responses = await asyncio.gather(*tasks)
        return list(responses)

    async def _process_single_request(self, request: BatchRequest):
        """
        Process a single request with semaphore-based rate limiting.

        Acquires the concurrency semaphore before making the API call,
        ensuring the maximum concurrent request limit is respected.

        Args:
            request: The BatchRequest to process.

        Returns:
            BatchResponse with the API result or error information.
        """
        # Acquire semaphore to respect concurrent request limits
        async with self._semaphore:
            # Update request status to in-progress
            request.status = RequestStatus.IN_PROGRESS
            # Log the request being processed
            logger.debug("Processing request id=%s", request.request_id)

            try:
                # Build the API payload from the request
                payload = self._request_builder.build_api_payload(request)
                # Send the request to the LLM API
                raw_response = await self._api_client.send_request(payload)
                # Parse the raw response into a structured model
                response = self._response_handler.parse_response(request.request_id, raw_response)
                # Update request status to completed
                request.status = RequestStatus.COMPLETED
                return response

            except ApiClientError as api_error:
                # Log the API failure
                logger.error("Request id=%s failed: %s", request.request_id, api_error)
                # Update request status to failed
                request.status = RequestStatus.FAILED
                # Return an error response
                return self._response_handler.parse_error_response(request.request_id, str(api_error))

    async def close(self) -> None:
        """
        Release all resources held by the batch processor.

        Closes the underlying API client HTTP connection pool.
        Should be called when batch processing is complete.
        """
        # Close the API client to release connections
        await self._api_client.close()
        # Log resource cleanup
        logger.info("BatchProcessor resources released")
