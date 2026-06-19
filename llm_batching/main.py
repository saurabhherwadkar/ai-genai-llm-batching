# main.py
# Entry point for the LLM batching demonstration application.
# Demonstrates how to use the batch processing pipeline to send
# multiple prompts efficiently to an LLM API endpoint.

import asyncio
import logging

from llm_batching.batch_processor import BatchProcessor
from llm_batching.config import create_config
from llm_batching.logger_setup import configure_logging

# Module-level logger for main application flow
logger = logging.getLogger(__name__)

# Example prompts demonstrating various LLM use cases for batching
EXAMPLE_PROMPTS = [
    "Explain what a neural network is in one sentence.",
    "What is the capital of France?",
    "Write a haiku about programming.",
    "Explain the difference between a list and a tuple in Python.",
    "What are the three laws of thermodynamics?",
    "Summarize the concept of machine learning in two sentences.",
    "What is the time complexity of binary search?",
    "Name three benefits of containerization with Docker.",
]


async def run_batch_processing() -> None:
    """
    Execute the full batch processing demonstration.

    Loads configuration, initializes the processor, sends example
    prompts through the pipeline, and displays the results.
    """
    # Load application configuration from YAML and environment
    config = create_config()

    # Configure logging with the level from configuration
    configure_logging(config.log_level)

    # Log the start of the demonstration
    logger.info("Starting LLM Batching demonstration")
    logger.info("Processing %d example prompts", len(EXAMPLE_PROMPTS))

    # Initialize the batch processor with configuration
    processor = BatchProcessor(config)

    try:
        # Process all example prompts through the batching pipeline
        result = await processor.process_prompts(EXAMPLE_PROMPTS)

        # Display the batch processing results
        _display_results(result)

    finally:
        # Ensure resources are released regardless of success or failure
        await processor.close()


def _display_results(result) -> None:
    """
    Display the batch processing results in a readable format.

    Prints summary statistics and individual response details
    to demonstrate the output of the batching pipeline.

    Args:
        result: The BatchResult containing all responses and statistics.
    """
    # Print summary header
    print("\n" + "=" * 60)
    print("BATCH PROCESSING RESULTS")
    print("=" * 60)

    # Print summary statistics
    print(f"Total Requests:  {result.total_requests}")
    print(f"Successful:      {result.successful_count}")
    print(f"Failed:          {result.failed_count}")
    print(f"Total Tokens:    {result.total_tokens_used}")
    print("-" * 60)

    # Print each individual response
    for response in result.responses:
        # Print the request ID and status
        status_indicator = "OK" if response.success else "FAIL"
        print(f"\n[{status_indicator}] Request: {response.request_id}")
        if response.success:
            # Print successful response content (truncated for display)
            content_preview = response.content[:100] + "..." if len(response.content) > 100 else response.content
            print(f"    Response: {content_preview}")
            print(f"    Tokens:   {response.tokens_used}")
        else:
            # Print error information for failed requests
            print(f"    Error:    {response.error_message}")

    # Print footer
    print("\n" + "=" * 60)


def main() -> None:
    """
    Application entry point that runs the async batch processing.

    Sets up the asyncio event loop and executes the batch
    processing demonstration coroutine.
    """
    # Run the async batch processing in the event loop
    asyncio.run(run_batch_processing())


# Execute main when this module is run directly
if __name__ == "__main__":
    main()
