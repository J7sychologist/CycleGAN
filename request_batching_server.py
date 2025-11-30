import sys
import asyncio
import itertools
import functools
from sanic import Sanic
from sanic.response import json, text
from sanic.log import logger
from sanic.exceptions import ServerError

import sanic
import threading
import PIL.Image
import io
import torch
import torchvision
from p2ch15.cyclegan import get_pretrained_model

app = Sanic(__name__)

device = torch.device("cpu")
MAX_QUEUE_SIZE = 3
MAX_BATCH_SIZE = 2
MAX_WAIT = 1


class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg


class ModelRunner:
    def __init__(self, model_name):
        self.model_name = model_name
        self.queue = []

        # Инициализируем здесь, а не в model_runner
        self.queue_lock = asyncio.Lock()
        self.needs_processing = asyncio.Event()
        self.needs_processing_timer = None

        self.model = get_pretrained_model(self.model_name, map_location=device)
        self.model_runner_started = False

    def schedule_processing_if_needed(self):
        if len(self.queue) >= MAX_BATCH_SIZE:
            logger.debug("next batch ready when processing a batch")
            self.needs_processing.set()
        elif self.queue:
            logger.debug("queue nonempty when processing a batch, setting next timer")
            self.needs_processing_timer = app.loop.call_at(
                self.queue[0]["time"] + MAX_WAIT, self.needs_processing.set
            )

    async def process_input(self, input):
        # Убедимся, что model_runner запущен
        if not self.model_runner_started:
            raise HandlingError("Model runner not ready", code=503)

        our_task = {
            "done_event": asyncio.Event(),
            "input": input,
            "time": app.loop.time(),
        }

        async with self.queue_lock:
            if len(self.queue) >= MAX_QUEUE_SIZE:
                raise HandlingError("I'm too busy", code=503)
            self.queue.append(our_task)
            logger.debug("enqueued task. new queue size {}".format(len(self.queue)))
            self.schedule_processing_if_needed()

        await our_task["done_event"].wait()
        return our_task["output"]

    def run_model(self, batch):
        return self.model(batch.to(device)).to("cpu")

    async def model_runner(self):
        self.model_runner_started = True
        logger.info("started model runner for {}".format(self.model_name))

        while True:
            await self.needs_processing.wait()
            self.needs_processing.clear()

            if self.needs_processing_timer is not None:
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None

            async with self.queue_lock:
                if self.queue:
                    longest_wait = app.loop.time() - self.queue[0]["time"]
                else:
                    longest_wait = None

                logger.debug(
                    "launching processing. queue size: {}. longest wait: {}".format(
                        len(self.queue), longest_wait
                    )
                )

                to_process = self.queue[:MAX_BATCH_SIZE]
                del self.queue[: len(to_process)]
                self.schedule_processing_if_needed()

            if to_process:
                batch = torch.stack([t["input"] for t in to_process], dim=0)

                result = await app.loop.run_in_executor(
                    None, functools.partial(self.run_model, batch)
                )

                for t, r in zip(to_process, result):
                    t["output"] = r
                    t["done_event"].set()

                del to_process


# Инициализация модели
if len(sys.argv) < 2:
    print("Usage: python request_batching_server.py <model_weights.pt>")
    sys.exit(1)

style_transfer_runner = ModelRunner(sys.argv[1])


@app.listener("before_server_start")
async def setup_model_runner(app, loop):
    # Запускаем model_runner до начала обработки запросов
    app.add_task(style_transfer_runner.model_runner())
    await asyncio.sleep(0.1)  # Даем время для инициализации


@app.route("/image", methods=["PUT"], stream=True)
async def image(request):
    try:
        print("Received request headers:", dict(request.headers))
        content_length = int(request.headers.get("content-length", "0"))
        MAX_SIZE = 2**22  # 10MB

        if content_length > MAX_SIZE:
            raise HandlingError("Too large")

        data = bytearray()
        while True:
            data_part = await request.stream.read()
            if data_part is None:
                break
            data.extend(data_part)
            if len(data) > MAX_SIZE:
                raise HandlingError("Too large")

        print(f"Received image data: {len(data)} bytes")

        # Обработка изображения
        im = PIL.Image.open(io.BytesIO(data))
        print(f"Original image size: {im.size}, mode: {im.mode}")

        im = torchvision.transforms.functional.resize(im, (480, 640))
        im = torchvision.transforms.functional.to_tensor(im)
        im = im[:3]  # drop alpha channel if present
        print(f"Processed tensor shape: {im.shape}")

        if im.dim() != 3 or im.size(0) < 3 or im.size(0) > 4:
            raise HandlingError("need rgb image")

        print("Processing image with model...")
        out_im = await style_transfer_runner.process_input(im)
        print("Model processing completed")

        out_im = torchvision.transforms.functional.to_pil_image(out_im)
        imgByteArr = io.BytesIO()
        out_im.save(imgByteArr, format="JPEG")

        result_size = imgByteArr.tell()
        print(f"Result image size: {result_size} bytes")

        return sanic.response.raw(
            imgByteArr.getvalue(), status=200, content_type="image/jpeg"
        )

    except HandlingError as e:
        print(f"HandlingError: {e.handling_msg}")
        return sanic.response.text(e.handling_msg, status=e.handling_code)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return sanic.response.text("Internal server error", status=500)


@app.route("/health", methods=["GET"])
async def health_check(request):
    return json(
        {"status": "healthy", "model_ready": style_transfer_runner.model_runner_started}
    )


@app.route("/ready", methods=["GET"])
async def ready_check(request):
    """Endpoint specifically for checking if model is ready"""
    if style_transfer_runner.model_runner_started:
        return json({"status": "ready"})
    else:
        return json({"status": "initializing"}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, workers=1, access_log=False)
