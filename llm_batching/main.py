import asyncio
import logging

from llm_batching.batch_queue import BatchQueue
from llm_batching.config import create_config
from llm_batching.logger_setup import configure_logging
from llm_batching.models import BatchResponse

logger = logging.getLogger(__name__)

EXAMPLE_PROMPTS = [
    "Explain what a neural network is in one sentence.",
    "What is the capital of France?",
    "Write a haiku about programming.",
    "Explain the difference between a list and a tuple in Python.",
    "What are the three laws of thermodynamics?",
    "Summarize the concept of machine learning in two sentences.",
    "What is the time complexity of binary search?",
    "Name three benefits of containerization with Docker.",
    "What is the CAP theorem?",
    "Explain recursion in one sentence.",
    "What is a hash table?",
    "Name three sorting algorithms and their time complexities.",
    "What is the difference between TCP and UDP?",
    "Explain what an API gateway does.",
    "What is eventual consistency?",
    "Describe the observer design pattern in one sentence.",
    "What is a deadlock?",
    "Explain the difference between concurrency and parallelism.",
    "What is memoization?",
    "Name three NoSQL database types.",
    "What is a load balancer?",
    "Explain what CORS is in one sentence.",
    "What is the difference between authentication and authorization?",
    "Describe the pub/sub messaging pattern.",
    "What is a circuit breaker pattern?",
]


async def run_batch_queue_demo() -> None:
    config = create_config()
    configure_logging(config.log_level)

    logger.info("Starting BatchQueue demonstration")
    logger.info(
        "Submitting %d prompts (queue_batch_size=%d, flush_timeout=%dms)",
        len(EXAMPLE_PROMPTS),
        config.queue_batch_size,
        config.queue_flush_timeout_ms,
    )

    queue = BatchQueue(config)
    await queue.start()

    try:
        # Simulate many independent callers submitting concurrently
        tasks = [queue.submit(prompt) for prompt in EXAMPLE_PROMPTS]
        responses: list[BatchResponse] = await asyncio.gather(*tasks)
        _display_results(responses)
    finally:
        await queue.stop()


def _display_results(responses: list[BatchResponse]) -> None:
    successful = [r for r in responses if r.success]
    failed = [r for r in responses if not r.success]
    total_tokens = sum(r.tokens_used for r in responses)

    print("\n" + "=" * 60)
    print("BATCH QUEUE RESULTS")
    print("=" * 60)
    print(f"Total Requests:  {len(responses)}")
    print(f"Successful:      {len(successful)}")
    print(f"Failed:          {len(failed)}")
    print(f"Total Tokens:    {total_tokens}")
    print("-" * 60)

    for response in responses:
        status_indicator = "OK" if response.success else "FAIL"
        print(f"\n[{status_indicator}] Request: {response.request_id}")
        if response.success:
            content_preview = response.content[:100] + "..." if len(response.content) > 100 else response.content
            print(f"    Response: {content_preview}")
            print(f"    Tokens:   {response.tokens_used}")
        else:
            print(f"    Error:    {response.error_message}")

    print("\n" + "=" * 60)


def main() -> None:
    asyncio.run(run_batch_queue_demo())


if __name__ == "__main__":
    main()
