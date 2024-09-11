import subprocess as sp
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_gpu_memory(mode: str = 'used', one_gpu: bool = True):
    """
    Get the current GPU memory usage. Might break if GPU is not available.
    :param mode: str, 'used', 'total' or 'free'
    :param one_gpu: bool, if True, return the memory usage of the first GPU
    :return: float or list of floats with the memory usage in GB.
    """
    assert mode in ['used', 'total', 'free']
    command = f"nvidia-smi --query-gpu=memory.{mode} --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    if one_gpu:
        memory_result = memory_free_values[0] / 1024
        logger.info(f"GPU memory {mode}: {memory_result} GB")
        return memory_result
    return [m / 1024 for m in memory_free_values]
