import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.solver import assembly, solver
from networks_fenicsx.config import Config
from networks_fenicsx.utils.timers import timing_dict

# Clear fenics cache
print('Clearing cache')
os.system('dijitso clean')

cfg = Config()
cfg.outdir = "demo_perf"
cfg.export = True


class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[1])


lcar = 0.5

# Cleaning directory only once
cfg.clean_dir()
cfg.clean = False

p = Path(cfg.outdir)
p.mkdir(exist_ok=True)

for n in range(2, 4):

    with (p / 'profiling.txt').open('a') as f:
        f.write("n: " + str(n) + "\n")

    # Create tree
    G = mesh_generation.make_tree(n=n, H=n, W=n, cfg=cfg)

    assembler = assembly.Assembler(cfg, G)
    assembler.compute_forms(p_bc_ex=p_bc_expr())
    assembler.assemble()

    solver_ = solver.Solver(cfg, G, assembler)
    (fluxes, global_flux, pressure) = solver_.solve()

t_dict = timing_dict(cfg.outdir)

fig, ax = plt.subplots()
ax.plot(t_dict["n"], t_dict["solve"])
plt.show()
