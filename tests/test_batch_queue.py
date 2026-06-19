import asyncio

import httpx
import pytest
import respx

from llm_batching.batch_queue import BatchQueue


class TestBatchQueue:
    """Tests for the queue-based batch aggregation."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_submit_single_prompt(self, test_config, sample_api_response):
        """A single submitted prompt gets a response."""
        respx.post(f"{test_config.api_base_url}/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_api_response)
        )
        queue = BatchQueue(test_config)
        await queue.start()
        try:
            response = await queue.submit("Hello")
            assert response.success
            assert response.content == "Python is a high-level programming language."
        finally:
            await queue.stop()

    @respx.mock
    @pytest.mark.asyncio
    async def test_batches_multiple_prompts(self, test_config, sample_api_response):
        """Multiple concurrent submits are batched together."""
        route = respx.post(f"{test_config.api_base_url}/chat/completions")
        route.mock(return_value=httpx.Response(200, json=sample_api_response))

        # Use a small batch size to force batching
        test_config.queue_batch_size = 3
        test_config.queue_flush_timeout_ms = 500

        queue = BatchQueue(test_config)
        await queue.start()
        try:
            tasks = [queue.submit(f"Prompt {i}") for i in range(6)]
            responses = await asyncio.gather(*tasks)
            assert len(responses) == 6
            assert all(r.success for r in responses)
        finally:
            await queue.stop()

    @respx.mock
    @pytest.mark.asyncio
    async def test_flush_on_timeout(self, test_config, sample_api_response):
        """Requests flush after timeout even if batch is not full."""
        respx.post(f"{test_config.api_base_url}/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_api_response)
        )
        test_config.queue_batch_size = 100  # very large — won't fill
        test_config.queue_flush_timeout_ms = 50  # short timeout forces flush

        queue = BatchQueue(test_config)
        await queue.start()
        try:
            response = await queue.submit("Only one prompt")
            assert response.success
        finally:
            await queue.stop()

    @respx.mock
    @pytest.mark.asyncio
    async def test_handles_api_failure(self, test_config):
        """Failed API calls result in error responses, not exceptions."""
        respx.post(f"{test_config.api_base_url}/chat/completions").mock(
            return_value=httpx.Response(400, json={"error": "Bad request"})
        )
        queue = BatchQueue(test_config)
        await queue.start()
        try:
            response = await queue.submit("Bad prompt")
            assert not response.success
            assert response.error_message != ""
        finally:
            await queue.stop()

    @respx.mock
    @pytest.mark.asyncio
    async def test_large_burst(self, test_config, sample_api_response):
        """A burst of many requests is handled without dropping any."""
        respx.post(f"{test_config.api_base_url}/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_api_response)
        )
        test_config.queue_batch_size = 5
        test_config.queue_flush_timeout_ms = 50

        queue = BatchQueue(test_config)
        await queue.start()
        try:
            tasks = [queue.submit(f"Prompt {i}") for i in range(25)]
            responses = await asyncio.gather(*tasks)
            assert len(responses) == 25
            assert all(r.success for r in responses)
        finally:
            await queue.stop()
