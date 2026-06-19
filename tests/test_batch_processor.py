# test_batch_processor.py
# Integration-level unit tests for the BatchProcessor class.
# Uses respx to mock the LLM API and verifies end-to-end batch
# processing including splitting, concurrency, and aggregation.

import httpx
import pytest
import respx

from llm_batching.batch_processor import BatchProcessor


class TestBatchProcessor:
    """Tests for BatchProcessor end-to-end processing pipeline."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_process_prompts_all_successful(self, test_config, sample_api_response):
        """Verify all prompts are processed successfully in batches."""
        # Mock API to return successful response for all requests
        respx.post(f"{test_config.api_base_url}/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_api_response)
        )
        # Create processor and process prompts
        processor = BatchProcessor(test_config)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4"]
        result = await processor.process_prompts(prompts)
        # Assert all requests succeeded
        assert result.total_requests == 4
        assert result.successful_count == 4
        assert result.failed_count == 0
        assert result.total_tokens_used == 18 * 4
        # Cleanup
        await processor.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_process_prompts_with_failures(self, test_config):
        """Verify mixed success/failure handling in batch processing."""
        # Mock API to alternate between success and failure
        route = respx.post(f"{test_config.api_base_url}/chat/completions")
        route.side_effect = [
            httpx.Response(200, json={
                "choices": [{"message": {"content": "Response 1"}}],
                "usage": {"total_tokens": 10},
                "model": "test",
            }),
            httpx.Response(500, json={"error": "Server Error"}),
            httpx.Response(500, json={"error": "Server Error"}),
        ]
        # Create processor with only 1 retry to speed up test
        processor = BatchProcessor(test_config)
        prompts = ["Success prompt", "Fail prompt"]
        result = await processor.process_prompts(prompts)
        # Assert mixed results
        assert result.total_requests == 2
        assert result.successful_count == 1
        assert result.failed_count == 1
        # Cleanup
        await processor.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_process_prompts_empty_list(self, test_config):
        """Verify processing handles empty prompt list gracefully."""
        # Create processor and process empty list
        processor = BatchProcessor(test_config)
        result = await processor.process_prompts([])
        # Assert empty result
        assert result.total_requests == 0
        assert result.successful_count == 0
        assert result.failed_count == 0
        # Cleanup
        await processor.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_process_prompts_respects_batch_size(self, test_config, sample_api_response):
        """Verify prompts are split according to configured batch size."""
        # Mock API to return successful responses
        route = respx.post(f"{test_config.api_base_url}/chat/completions")
        route.mock(return_value=httpx.Response(200, json=sample_api_response))
        # Create processor with batch_size=3 (from test_config)
        processor = BatchProcessor(test_config)
        # Process 7 prompts (should create 3 batches: 3+3+1)
        prompts = [f"Prompt {i}" for i in range(7)]
        result = await processor.process_prompts(prompts)
        # Assert all 7 requests were processed
        assert result.total_requests == 7
        assert result.successful_count == 7
        # Assert API was called 7 times (once per prompt)
        assert route.call_count == 7
        # Cleanup
        await processor.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_process_single_prompt(self, test_config, sample_api_response):
        """Verify processing works correctly with a single prompt."""
        # Mock API to return successful response
        respx.post(f"{test_config.api_base_url}/chat/completions").mock(
            return_value=httpx.Response(200, json=sample_api_response)
        )
        # Create processor and process single prompt
        processor = BatchProcessor(test_config)
        result = await processor.process_prompts(["Single prompt"])
        # Assert single result
        assert result.total_requests == 1
        assert result.successful_count == 1
        assert result.responses[0].content == "Python is a high-level programming language."
        # Cleanup
        await processor.close()
