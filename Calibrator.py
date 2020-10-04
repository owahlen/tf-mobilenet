# Calibrator.py
import os
import logging
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

class _Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, opt_shape=(1,160,160,3)):
        super().__init__()

        self._batch_size = opt_shape[0]
        num_samples = 1000
        self.batches = ((np.random.random(opt_shape[1:])*2.0-1.0).astype(np.float32) for i in range(num_samples))
        self.device_input = cuda.mem_alloc(np.zeros(opt_shape, dtype=np.float32).nbytes)
        self.cache_file = "calibration.cache"

    def get_batch(self, names, p_str=None):
        try:
            batch = next(self.batches)
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]
        except StopIteration:
            return None

    def get_batch_size(self):
        return self._batch_size

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)