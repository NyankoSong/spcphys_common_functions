
import os


def _determine_processes(num_processes) -> int:
    """Determine the number of processes to use for parallel computation.

    :param num_processes: Number of processes to use. Can be None, a fraction (<1), or an integer (>=1)
    :type num_processes: int, float, or None
    :return: The number of processes to use
    :rtype: int
    
    If num_processes is None, use 50% of available CPU cores.
    If num_processes < 1, use that fraction of available CPU cores.
    If num_processes >= 1, use that many CPU cores.
    At least one process is always used.
    """
    if num_processes is None:
        num_processes = int(os.cpu_count() * 0.5)
    elif num_processes < 1 and num_processes > 0:
        num_processes = int(os.cpu_count() * num_processes)
    elif num_processes > 1:
        num_processes = int(num_processes)
        
    if num_processes < 1:
        num_processes = 1
        
    return num_processes