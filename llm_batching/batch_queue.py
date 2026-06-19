import asyncio
import logging

from llm_batching.api_client import ApiClient, ApiClientError
from llm_batching.config import AppConfig
from llm_batching.models import BatchResponse, RequestStatus
from llm_batching.request_builder import RequestBuilder
from llm_batching.response_handler import ResponseHandler

logger = logging.getLogger(__name__)


class BatchQueue:
    """
    Aggregates incoming LLM requests and dispatches them in batches.

    Callers submit individual prompts via `submit()` and receive a future
    that resolves when their specific request completes. A background worker
    collects requests and flushes them as a batch when either the batch size
    is reached or the flush timeout elapses — whichever comes first.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._request_builder = RequestBuilder(config)
        self._response_handler = ResponseHandler()
        self._api_client = ApiClient(config)
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        self._queue: asyncio.Queue[tuple[str, asyncio.Future]] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._running = False

        logger.info(
            "BatchQueue initialized: queue_batch_size=%d, queue_flush_timeout_ms=%d",
            config.queue_batch_size,
            config.queue_flush_timeout_ms,
        )

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("BatchQueue worker started")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        await self._api_client.close()
        logger.info("BatchQueue stopped and resources released")

    async def submit(self, prompt: str) -> BatchResponse:
        """
        Submit a single prompt and wait for its response.

        The prompt is enqueued and will be dispatched as part of the next
        batch. Returns the BatchResponse once the batch containing this
        request has been processed.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[BatchResponse] = loop.create_future()
        await self._queue.put((prompt, future))
        logger.debug("Prompt submitted, queue_size=%d", self._queue.qsize())
        return await future

    async def _worker_loop(self) -> None:
        batch_size = self._config.queue_batch_size
        flush_timeout = self._config.queue_flush_timeout_ms / 1000.0

        while self._running:
            pending: list[tuple[str, asyncio.Future]] = []

            # Wait for the first item (blocks until something arrives)
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                pending.append(item)
            except asyncio.TimeoutError:
                continue

            # Collect more items up to batch_size or until timeout
            deadline = asyncio.get_event_loop().time() + flush_timeout
            while len(pending) < batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    pending.append(item)
                except asyncio.TimeoutError:
                    break

            if pending:
                logger.info("Flushing batch of %d requests", len(pending))
                await self._dispatch_batch(pending)

    async def _dispatch_batch(self, pending: list[tuple[str, asyncio.Future]]) -> None:
        requests = []
        futures = []
        for prompt, future in pending:
            req = self._request_builder.create_request(prompt)
            req.status = RequestStatus.IN_PROGRESS
            requests.append(req)
            futures.append(future)

        tasks = [self._process_single(req) for req in requests]
        responses = await asyncio.gather(*tasks)

        for future, response in zip(futures, responses):
            if not future.cancelled():
                future.set_result(response)

    async def _process_single(self, request) -> BatchResponse:
        async with self._semaphore:
            try:
                payload = self._request_builder.build_api_payload(request)
                raw_response = await self._api_client.send_request(payload)
                response = self._response_handler.parse_response(request.request_id, raw_response)
                request.status = RequestStatus.COMPLETED
                return response
            except ApiClientError as e:
                logger.error("Request id=%s failed: %s", request.request_id, e)
                request.status = RequestStatus.FAILED
                return self._response_handler.parse_error_response(request.request_id, str(e))
