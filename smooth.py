import trimesh
import numpy as np

mesh = trimesh.load('outs/7b00e029725c0c96473f10e6caaeca56_0315_4w5121l/mesh/40000_mesh.obj')
mesh = trimesh.smoothing.filter_laplacian(mesh)
mesh.export('outs/7b00e029725c0c96473f10e6caaeca56_0315_4w5121l/mesh/40000_smooth_mesh.obj')