import networkx as nx
import numpy as np
from typing import List
import copy

from mpi4py import MPI
from dolfinx import fem, io, mesh

import gmsh

from networks_fenicsx.utils.timers import timeit
from networks_fenicsx import config

'''
The Graphnics class constructs fenics meshes from networkx directed graphs.

'''


class NetworkGraph(nx.DiGraph):
    '''
    Make FEniCSx mesh from networkx directed graph
    '''

    def __init__(self, config: config.Config):
        nx.DiGraph.__init__(self)

        self.comm = MPI.COMM_WORLD
        self.cfg = config
        self.cfg.clean_dir()

        self.bifurcation_ixs: List[int] = []  # noqa: F821
        self.boundary_ixs: List[int] = []  # noqa: F821

        self.msh = None
        self.subdomains = None
        self.boundaries = None
        self.global_tangent = None

        self.BIF_IN = 1
        self.BIF_OUT = 2
        self.BOUN_IN = 3
        self.BOUN_OUT = 4

    @timeit
    def build_mesh(self):
        '''
        Makes a fenics mesh from the graph
        Args:
            lcar (float): Characteristic length of the elements
        Returns:
            mesh : the global mesh
        '''

        self.geom_dim = len(self.nodes[1]['pos'])
        self.num_edges = len(self.edges)

        vertex_coords = np.asarray([self.nodes[v]['pos'] for v in self.nodes()])
        # print("coords = ", vertex_coords)
        cells_array = np.asarray([[u, v] for u, v in self.edges()])

        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 1)
        pts = []
        lines = []
        for i, v in enumerate(vertex_coords):
            pts.append(gmsh.model.geo.addPoint(v[0], v[1], v[2], self.cfg.lcar))

        for i, c in enumerate(cells_array):
            lines.append(gmsh.model.geo.addLine(pts[c[0]], pts[c[1]]))

        gmsh.model.geo.synchronize()
        for i, line in enumerate(lines):
            gmsh.model.addPhysicalGroup(1, [line], i)

        gmsh.model.mesh.generate(1)

        self.msh, self.subdomains, self.boundaries = io.gmshio.model_to_mesh(
            gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3)
        gmsh.finalize()

        if MPI.COMM_WORLD.rank == 0:
            print("Nb vertices = ", self.msh.topology.index_map(0).size_global)
        if self.cfg.export:
            with io.XDMFFile(self.comm, self.cfg.outdir + "/mesh/mesh.xdmf", "w") as file:
                file.write_mesh(self.msh)
                file.write_meshtags(self.subdomains)

    @timeit
    def build_network_submeshes(self):

        for i, (u, v) in enumerate(self.edges):
            edge_subdomain = self.subdomains.find(i)

            self.edges[u, v]['submesh'], self.edges[u, v]['entity_map'] = mesh.create_submesh(self.msh, self.msh.topology.dim, edge_subdomain)[0:2]
            self.edges[u, v]['tag'] = i

            self.edges[u, v]["entities"] = []
            self.edges[u, v]["b_values"] = []

    @timeit
    def build_markers(self):
        # Marking the bifurcations (in/out) and boundaries (in/out) for extermities of each edges
        for n, v in enumerate(self.nodes()):
            num_conn_edges = len(self.in_edges(v)) + len(self.out_edges(v))
            if num_conn_edges == 0:
                print(f'Node {v} in G is lonely (i.e. unconnected)')

            bifurcation = bool(num_conn_edges > 1)
            boundary = bool(num_conn_edges == 1)
            if bifurcation:
                self.bifurcation_ixs.append(v)
            if boundary:
                self.boundary_ixs.append(v)

            for i, e in enumerate(self.in_edges(v)):
                e_msh = self.edges[e]['submesh']
                entities = mesh.locate_entities(e_msh, 0, lambda x: np.logical_and(np.isclose(x[0], self.nodes[v]['pos'][0]),
                                                                                   np.isclose(x[1], self.nodes[v]['pos'][1]),
                                                                                   np.isclose(x[2], self.nodes[v]['pos'][2])))
                self.edges[e]["entities"].append(entities)
                if bifurcation:
                    b_values_in = np.full(entities.shape, self.BIF_IN, np.intc)
                elif boundary:
                    b_values_in = np.full(entities.shape, self.BOUN_OUT, np.intc)
                self.edges[e]["b_values"].append(b_values_in)

            for i, e in enumerate(self.out_edges(v)):
                e_msh = self.edges[e]['submesh']
                entities = mesh.locate_entities(e_msh, 0, lambda x: np.logical_and(np.isclose(x[0], self.nodes[v]['pos'][0]),
                                                                                   np.isclose(x[1], self.nodes[v]['pos'][1]),
                                                                                   np.isclose(x[2], self.nodes[v]['pos'][2])))
                self.edges[e]["entities"].append(entities)
                if bifurcation:
                    b_values_out = np.full(entities.shape, self.BIF_OUT, np.intc)
                elif boundary:
                    b_values_out = np.full(entities.shape, self.BOUN_IN, np.intc)
                self.edges[e]["b_values"].append(b_values_out)

        # Creating the edges meshtags
        for i, e in enumerate(self.edges):
            e_msh = self.edges[e]['submesh']
            indices, pos = np.unique(np.hstack(self.edges[e]["entities"]), return_index=True)
            self.edges[e]['vf'] = mesh.meshtags(e_msh, 0, indices, np.hstack(self.edges[e]["b_values"])[pos])
            e_msh.topology.create_connectivity(0, 1)
            # print("vf (", i , ")= ", self.edges[e]['vf'].values)

            if self.cfg.export:
                with io.XDMFFile(self.comm, self.cfg.outdir + "/mesh/edge_" + str(i) + ".xdmf", "w") as file:
                    file.write_mesh(e_msh)
                    file.write_meshtags(self.edges[e]['vf'])

    @timeit
    def compute_tangent(self):

        gdim = self.msh.geometry.dim

        boun_in = []
        boun_out = []

        DG0 = fem.VectorFunctionSpace(self.msh, ("DG", 0), dim=self.msh.geometry.dim)
        self.global_tangent = fem.Function(DG0)

        for i, (u, v) in enumerate(self.edges):
            edge_subdomain = self.subdomains.find(i)

            tangent = np.asarray(self.nodes[v]['pos']) - np.asarray(self.nodes[u]['pos'])
            tangent = tangent * 1 / np.linalg.norm(tangent)
            self.edges[u, v]['tangent'] = tangent

            # Finding BOUN_IN and BOUN_OUT dofs coordinates
            P1_e = fem.FunctionSpace(self.edges[(u, v)]['submesh'], ("Lagrange", 1))
            dof_coords = P1_e.tabulate_dof_coordinates()

            if self.edges[(u, v)]["vf"].find(self.BOUN_IN):
                boun_in.append(dof_coords[self.edges[(u, v)]["vf"].find(self.BOUN_IN)])
            if self.edges[(u, v)]["vf"].find(self.BOUN_OUT):
                boun_out.append(dof_coords[self.edges[(u, v)]["vf"].find(self.BOUN_OUT)])

        assert len(boun_in) > 0 and len(boun_out) > 0, \
            "Error in submeshes markers : Need at least one inlet and one outlet"
        global_dir = boun_out[0][0] - boun_in[0][0]
        global_dir[0] = 0  # Global tangent oriented in the y direction
        global_dir_copy = copy.deepcopy(global_dir)

        for i, (u, v) in enumerate(self.edges):
            tangent = self.edges[u, v]['tangent']
            t_dot_glob_dir = np.dot(tangent, global_dir)
            while t_dot_glob_dir == 0:  # if global_dir is perpendicular to tangent
                global_dir_copy[0] += 1
                t_dot_glob_dir = np.dot(tangent, global_dir_copy)
            global_dir_correction = t_dot_glob_dir * 1 / np.linalg.norm(t_dot_glob_dir)

            # Update tangent with corrected direction
            self.edges[u, v]['tangent'] *= global_dir_correction

            edge_subdomain = self.subdomains.find(i)
            for cell in edge_subdomain:
                self.global_tangent.x.array[gdim * cell:gdim * (cell + 1)] = self.edges[u, v]['tangent']
        self.global_tangent.x.scatter_forward()
        print("global tg = ", self.global_tangent.x.array)

    def mesh(self):
        return self.msh

    def submeshes(self):
        return list(nx.get_edge_attributes(self, 'submesh').values())

    def tangent(self):

        if self.cfg.export:
            self.global_tangent.x.scatter_forward()
            with io.XDMFFile(self.comm, self.cfg.outdir + "/mesh/tangent.xdmf", "w") as file:
                file.write_mesh(self.msh)
                file.write_function(self.global_tangent)

        return self.global_tangent
