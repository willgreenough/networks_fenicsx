import os
import numpy as np
from pathlib import Path
from mpi4py import MPI

from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.solver import assembly, solver
from networks_fenicsx.config import Config
from networks_fenicsx.utils.timers import timing_dict, timing_table
from networks_fenicsx.utils.post_processing import export  # , perf_plot

cfg = Config()
cfg.outdir = "demo_perf"
cfg.export = True

cfg.flux_degree = 2
cfg.pressure_degree = 1


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[1])


# One element per segment
cfg.lcar = 2.0

# Cleaning directory only once
cfg.clean_dir()
cfg.clean = False

p = Path(cfg.outdir)
p.mkdir(exist_ok=True)

for n in range(5, 6):

    if MPI.COMM_WORLD.rank == 0:
        print('Clearing cache')
        os.system('rm -rf $HOME/.cache/fenics/')

        with (p / 'profiling.txt').open('a') as f:
            f.write("n: " + str(n) + "\n")

    # Create tree
    G = mesh_generation.make_tree(n=n, H=n, W=n, cfg=cfg)

    assembler = assembly.Assembler(cfg, G)
    # Compute forms
    assembler.compute_forms(p_bc_ex=p_bc_expr())
    # Assemble
    assembler.assemble()
    # Solve
    solver_ = solver.Solver(cfg, G, assembler)
    sol = solver_.solve()
    (fluxes, global_flux, pressure) = export(cfg, G, assembler.function_spaces, sol,
                                             export_dir="n" + str(n))
    # for i, flux in enumerate(fluxes):
    #     print("flux[", i, "] = ", flux.x.array)
    # print("pressure  = ", pressure.x.array)
    # print("q mean = ", np.mean(global_flux.x.array))

t_dict = timing_dict(cfg.outdir)
timing_table(cfg.outdir)
# perf_plot(t_dict)

print("n = ", t_dict["n"])
print("compute_forms time = ", t_dict["compute_forms"])
print("assembly time = ", t_dict["assemble"])
print("solve time = ", t_dict["solve"])
