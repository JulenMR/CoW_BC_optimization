import pyvista as pv
import numpy as np
import os
import matplotlib.pyplot as plt
import re 
import pandas as pd
import vtk
import glob
from vtkmodules.util.numpy_support import vtk_to_numpy

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def extract_3d_results_using_vtp(vtu_folder, vtp_outlet_path, time_step=0.01):

    cap_mask = pv.read(vtp_outlet_path)
    
    all_files = [f for f in os.listdir(vtu_folder) if f.endswith('.vtu')]
    all_files.sort(key=natural_sort_key)

    vtu_files = [os.path.join(vtu_folder, f) for f in all_files]
    times, pressures, flows = [], [], []

    for i, vtu_path in enumerate(vtu_files):
        mesh_vtu = pv.read(vtu_path)

        cap_results = cap_mask.sample(mesh_vtu)
        
        p_avg = np.mean(cap_results.point_data['Pressure'])
        p_avg_mmhg = p_avg/133.3
        pressures.append(p_avg_mmhg)

        cap_results = cap_results.compute_normals(cell_normals=True, point_normals=False)
        
        cap_cells = cap_results.point_data_to_cell_data()
        areas = cap_cells.compute_cell_sizes()['Area']
        vel_vectors = cap_cells.cell_data['Velocity']
        normals = cap_cells.cell_data['Normals']
        
        vel_normal_comp = np.einsum('ij,ij->i', vel_vectors, normals)
        flow = np.sum(vel_normal_comp * areas)
        if np.mean(flow) < 0:
            flow = -flow
        
        flows.append(flow)
        times.append(i * time_step)
    times = np.array(times)
    pressures = np.array(pressures)
    flow = np.array(flows)

    return times, pressures, flow 


surfaces_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Simulations/fine/8_asl/mesh-complete/mesh-surfaces/"
simulation_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Simulations/fine/8_asl/96-procs"
cap_files = glob.glob(os.path.join(surfaces_dir, "cap_*.vtp"))

fig_flow, ax_flow = plt.subplots(figsize=(10, 6))
fig_pres, ax_pres = plt.subplots(figsize=(10, 6))

for cap_path in cap_files:
    cap_name = os.path.basename(cap_path).replace('cap_', '').replace('.vtp', '')
    print(f"Procssing: {cap_name}...")
    
    times, pressures, flow = extract_3d_results_using_vtp(
        vtu_folder=simulation_file, 
        vtp_outlet_path=cap_path, 
        time_step=0.0218
    )
    flow_ml_s = flow / 1000.0
    pressure_mmhg = pressures / 1333.3
    
    ax_flow.plot(times, flow_ml_s, label=f"{cap_name}")
    ax_pres.plot(times, pressures, label=f"{cap_name}")

ax_flow.set_title(" PACS008 3D Flow results" )
ax_flow.set_xlabel("Time (s)")
ax_flow.set_ylabel("Flow (mL/s)")
ax_flow.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
ax_flow.grid(True, alpha=0.3)
fig_flow.savefig(os.path.join(simulation_file, "3D_flow_results.png"), dpi=300, bbox_inches='tight')

ax_pres.set_title("PACS008 3D Pressure results")
ax_pres.set_xlabel("Time (s)")
ax_pres.set_ylabel("Pressure (mmHg)")
ax_pres.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
ax_pres.grid(True, alpha=0.3)
fig_pres.savefig(os.path.join(simulation_file, "3D_pressure_results.png"), dpi=300, bbox_inches='tight')

print("Both graphs were saved in {simulation_file}")
plt.show()