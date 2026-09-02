import cv2
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)


class StreamCapture:
    """
    Threaded video capture for a single camera stream.

    Reconnects automatically on disconnect. Always exposes the latest frame
    via read() — stale frames are dropped so inference never blocks on IO.
    """

    def __init__(self, cam_id: int, url: str, queue_size: int = 2):
        self.cam_id = cam_id
        # Numeric string → webcam index; anything else → URL / file path
        self._source = int(url) if url.isdigit() else url
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"capture-{cam_id}"
        )

    def start(self) -> "StreamCapture":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stopped.is_set():
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                logger.warning(
                    "CAM-%02d: cannot open %s — retrying in 2 s", self.cam_id, self._source
                )
                time.sleep(2)
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info("CAM-%02d: connected to %s", self.cam_id, self._source)

            while not self._stopped.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("CAM-%02d: lost connection — reconnecting", self.cam_id)
                    break

                # Drop oldest frame to make room for the newest
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                self._queue.put((time.monotonic(), frame))

            cap.release()

    def read(self) -> tuple[float, any] | None:
        """Returns (monotonic_timestamp, frame) or None if no frame is ready."""
        try:
            return self._queue.get(timeout=0.05)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=3)
