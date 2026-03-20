import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy


import pyvista as pv

file = "/home/julenmr/Documents/CMU/Automatic_BC/CoW_Centerline_Data/cow_graphs/topcow_ct_001.vtp"
# Cargamos el grafo de TopCoW
mesh = pv.read(file)

# Encontramos puntos que son compartidos por más de 2 líneas (Bifurcaciones)
# Analizamos la conectividad de las celdas
def find_junctions(mesh):
    junction_nodes = []
    for i in range(mesh.n_points):
        # Buscamos cuántas líneas (cells) usan este punto
        connected_cells = mesh.extract_points([i]).n_cells
        if connected_cells > 2:
            junction_nodes.append(i)
    return junction_nodes


junctions = find_junctions(mesh)
print(f"Nodos para las líneas JUNCTION en el script 1D: {junctions}")

print("# NODOS DE JUNCTIONS")
for j_id in junctions:
    point = mesh.points[j_id]
    # point[0] es X, point[1] es Y, point[2] es Z
    print(f"NODE {j_id} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")

def write_1d_input(output_path, model_name, points, r_start, r_end, inflow_data, rcr_values, n_timesteps, timestep_size, save_every):
    # Nodes
    inlet_node = points[0]
    outlet_node = points[-1]
    
    nodes_txt = f"NODE 0 {inlet_node[0]:.6f} {inlet_node[1]:.6f} {inlet_node[2]:.6f}\n"
    nodes_txt += f"NODE 1 {outlet_node[0]:.6f} {outlet_node[1]:.6f} {outlet_node[2]:.6f}\n"
    
    # Calculate vessel length
    diffs = np.diff(points, axis=0)
    segments = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segments)
    
    # Area
    area_in = np.pi * (r_start**2)
    area_out = np.pi * (r_end**2)
    
    # Number of elements
    n_elements = max(20, int(total_length * 1.2))
    # Create a segment
    segments_txt = f"SEGMENT seg_0 0 {total_length:.6f} {n_elements} 0 1 {area_in:.6f} {area_out:.6f} 0.0 MAT1 NONE 0.0 0 0 RCR RCR_0\n"

    # Generate template
    full_content = f"""# 1D Simplified Model
MODEL {model_name}

{nodes_txt}

{segments_txt}

DATATABLE RCR_0 LIST
0.0 {rcr_values[0]}
0.0 {rcr_values[1]}
0.0 {rcr_values[2]}
0.0 0.0
ENDDATATABLE

DATATABLE INFLOW LIST
{inflow_data}
ENDDATATABLE

SOLVEROPTIONS {timestep_size} {save_every} {n_timesteps} 2 INFLOW FLOW 1.0e-5 1 1
MATERIAL MAT1 LINEAR 1.06 0.04 0.0 1.0 1.0e7 0.0 0.0
OUTPUT TEXT
"""
    path_1d = os.path.join(output_path, "ROM_1d")
    os.makedirs(path_1d, exist_ok=True)
    with open(os.path.join(path_1d, f"{model_name}.in"), "w") as f:
        f.write(full_content)
        