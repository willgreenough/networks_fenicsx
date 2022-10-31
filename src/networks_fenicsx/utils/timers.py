from time import perf_counter
from pathlib import Path
from functools import wraps


def timeit(func):
    """
    Prints and saves to 'profiling.txt' the execution time of func
    Args:
        func: function to time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        time_info = f'{func.__name__} executed in {end - start:.3f} seconds \n'

        # Write to profiling file
        p = Path(args[0].cfg.outdir)
        p.mkdir(exist_ok=True)
        with (p / 'profiling.txt').open('a') as f:
            f.write(time_info)

        return result

    return wrapper
