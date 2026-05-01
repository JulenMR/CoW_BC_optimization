import pyvista as pv
import numpy as np
import os
import matplotlib.pyplot as plt
import re 
import pandas as pd
import vtk
import glob
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib.cm as cm

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def extract_3d_results_using_vtp(vtu_folder, vtp_outlet_path, time_step_size=0.01, saved_every = 20):

    cap_mask = pv.read(vtp_outlet_path)
    
    all_files = [f for f in os.listdir(vtu_folder) if f.endswith('.vtu')]
    all_files.sort(key=natural_sort_key)

    vtu_files = [os.path.join(vtu_folder, f) for f in all_files]
    times, pressures, flows = [], [], []

    for i, vtu_path in enumerate(vtu_files):
        mesh_vtu = pv.read(vtu_path)
        time_step = time_step_size*saved_every
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

def reorder_legend(ax, order):

    handles, labels = ax.get_legend_handles_labels()
    dict_hl = dict(zip(labels, handles))
    
    ordered_handles = [dict_hl[name] for name in order if name in dict_hl]
    ordered_labels = [name for name in order if name in dict_hl]
    
    return ordered_handles, ordered_labels

surfaces_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-005/Simulations/fine/5_zeroD_opt/mesh-complete/mesh-surfaces/"
simulation_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-005/Simulations/fine/5_zeroD_opt/122-procs/"
cap_files = glob.glob(os.path.join(surfaces_dir, "cap_*.vtp"))

fig_f_in, ax_f_in = plt.subplots(figsize=(10, 6))
fig_f_out, ax_f_out = plt.subplots(figsize=(10, 6))
fig_p_in, ax_p_in = plt.subplots(figsize=(10, 6))
fig_p_out, ax_p_out = plt.subplots(figsize=(10, 6))

inlets = ["L_ICA", "R_ICA", "L_VA", "R_VA"]
outlets = ["L_SCA", "R_SCA", "L_PCA", "R_PCA", "L_MCA", "R_MCA", "L_ACA", "R_ACA"]

cap_names = sorted(inlets + outlets)
color_list = cm.get_cmap('tab10')(np.linspace(0, 1, len(cap_names)))
COLOR_MAP = dict(zip(cap_names, color_list))
DEFAULT_COLOR = "yellow"

for cap_path in cap_files:
    cap_name = os.path.basename(cap_path).replace('cap_', '').replace('.vtp', '')
    print(f"Processing: {cap_name}...")
    
    times, pressures, flow = extract_3d_results_using_vtp(
        vtu_folder=simulation_file, 
        vtp_outlet_path=cap_path, 
        time_step_size=0.001022,
        saved_every=20
    )
    flow_ml_s = flow / 1000.0
    line_color = COLOR_MAP.get(cap_name, DEFAULT_COLOR)

    if cap_name in inlets:
        ax_f_in.plot(times, flow_ml_s, label=cap_name, color=line_color, linewidth=2)
        ax_p_in.plot(times, pressures, label=cap_name, color=line_color, linewidth=2)
    elif cap_name in outlets:
        ax_f_out.plot(times, flow_ml_s, label=cap_name, color=line_color, linewidth=2)
        ax_p_out.plot(times, pressures, label=cap_name, color=line_color, linewidth=2)

plots_info = [
    (fig_f_in, ax_f_in, "Inlet Flow", "Flow (mL/s)", "3D_flow_inlets.png"),
    (fig_f_out, ax_f_out, "Outlet Flow", "Flow (mL/s)", "3D_flow_outlets.png"),
    (fig_p_in, ax_p_in, "Inlet Pressure", "Pressure (mmHg)", "3D_pres_inlets.png"),
    (fig_p_out, ax_p_out, "Outlet Pressure", "Pressure (mmHg)", "3D_pres_outlets.png")
]

for fig, ax, title, ylabel, filename in plots_info:
    ax.set_title(f"PACS005 {title} results")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    h, l = reorder_legend(ax, cap_names)
    if h: 
        ax.legend(h, l, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    fig.savefig(os.path.join(simulation_file, filename), dpi=300, bbox_inches='tight')

print(f"All 4 graphs saved in {simulation_file}")
plt.show()