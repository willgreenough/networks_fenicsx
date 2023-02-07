from dataclasses import dataclass
from pathlib import Path
from mpi4py import MPI
import shutil


@dataclass
class Config:
    outdir: str = "results"
    lcar: float = 1.0
    flux_degree: int = 2
    pressure_degree: int = 1
    export: bool = False
    clean: bool = True

    def clean_dir(self):
        if self.clean and MPI.COMM_WORLD.rank == 0:
            dirpath = Path(self.outdir)
            if dirpath.exists() and dirpath.is_dir():
                shutil.rmtree(dirpath, ignore_errors=True)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def default_parameters():
    return {k: v for k, v in Config.__dict__.items() if not k.startswith("_")}
