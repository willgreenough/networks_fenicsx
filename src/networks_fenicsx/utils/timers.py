from time import perf_counter
from pathlib import Path
from functools import wraps
from typing import Dict, List
from mpi4py import MPI

import pandas as pd


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
        time_info = f'{func.__name__}: {end - start:.3f} s \n'

        # Write to profiling file
        p = Path(args[0].cfg.outdir)
        p.mkdir(exist_ok=True)
        with (p / 'profiling.txt').open('a') as f:
            f.write(time_info)

        return result

    return wrapper


def timing_dict(outdir_path: str):
    """
    Read 'profiling.txt' and create a dictionary out of it
    Args:
       str : outdir path
    """
    p = Path(outdir_path)
    timing_file = (p / 'profiling.txt').open('r')
    timing_dict: Dict[str, List[float]] = dict()

    for line in timing_file:
        split_line = line.strip().split(':')
        keyword = split_line[0]
        value = float(split_line[1].split()[0])

        if keyword in timing_dict.keys():
            timing_dict[keyword].append(value)
        else:
            timing_dict[keyword] = [value]

    return timing_dict


def timing_table(outdir_path: str):
    """
    Read 'profiling.txt' and create a table data file
    Args:
       str : outdir path
    """
    t_dict = timing_dict(outdir_path)

    if MPI.COMM_WORLD.rank == 0:
        df = pd.DataFrame({
            'n': t_dict["n"],
            'forms': t_dict["compute_forms"],
            'assembly': t_dict["assemble"],
            'solve': t_dict["solve"]})

        df.to_csv(outdir_path + '/timings.txt', sep='\t', index=False)
