from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass
class Config:
    outdir: str = "results"
    lcar: float = 1.0
    export: bool = False
    clean: bool = True

    def clean_dir(self):
        if self.clean:
            print("Cleaning repo")
            dirpath = Path(self.outdir)
            print("dirpath = ", dirpath)
            if dirpath.exists() and dirpath.is_dir():
                print("rm ", dirpath)
                shutil.rmtree(dirpath)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def default_parameters():
    return {k: v for k, v in Config.__dict__.items() if not k.startswith("_")}
