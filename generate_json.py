import vtk
from vtk.util import numpy_support
import numpy as np
import json
import os
from collections import deque, defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def read_flow_file(path):
    if path and os.path.exists(path):
        data = np.loadtxt(path)
        return {
            "t": data[:, 0].tolist(), 
            "Q": np.abs(data[:, 1]).tolist()
        }
    return {"t": [0.0, 1.0], "Q": [0.5, 0.5]}

def generate_0d_json_multi_inlet(vtp_path, output_path, viscosity_val = 0.004, flow_files=None, rcr_values=None, tau = None):

    if not os.path.exists(vtp_path):
        print(f"Error: File not found at {vtp_path}")
        return

    print(f"\n" + "="*65)
    print(f"POSTPROCESSING CENTERLINE: {os.path.basename(vtp_path)}")
    print("="*65)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    # PHASE 1: EXTRACTION 
    branch_ids = numpy_support.vtk_to_numpy(polydata.GetCellData().GetArray("BranchID"))
    usage_tags = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("UsageTag"))
    radii = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("MaximumInscribedSphereRadius"))

    pos_to_node = {} 
    next_node_id = 0
    def get_node(pt_id):
        nonlocal next_node_id
        coords = polydata.GetPoint(pt_id)
        key = tuple(np.round(coords, 3)) 
        if key not in pos_to_node:
            pos_to_node[key] = next_node_id
            next_node_id += 1
        return pos_to_node[key]

    raw_branches = {} 
    node_to_branches = defaultdict(list) 
    inlet_nodes, outlet_nodes = set(), set()

    for i in range(polydata.GetNumberOfCells()): 
        cell = polydata.GetCell(i)
        b_id = int(branch_ids[i])
        pt_ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())] 
        n_start, n_end = get_node(pt_ids[0]), get_node(pt_ids[-1]) 
        raw_branches[b_id] = {'nodes': [n_start, n_end], 'pts': pt_ids, 'oriented': False} 
        node_to_branches[n_start].append(b_id)
        node_to_branches[n_end].append(b_id)
        for p_id in [pt_ids[0], pt_ids[-1]]:
            tag = int(usage_tags[p_id])
            if tag == 1: inlet_nodes.add(get_node(p_id)) 
            elif tag == 2: outlet_nodes.add(get_node(p_id)) 

    # PHASE 2: ORIENTATION & DEBUGGING GEOMETRY
    final_segments = {}
    queue = deque(list(inlet_nodes)) 
    
    while queue:
        curr_node = queue.popleft() 
        for b_id in node_to_branches[curr_node]: 
            br = raw_branches[b_id]
            if not br['oriented']: 
                n_in = curr_node 
                n_out = br['nodes'][1] if br['nodes'][0] == curr_node else br['nodes'][0] 
                coords_pts = [np.array(polydata.GetPoint(p)) for p in br['pts']]
                length = sum(np.linalg.norm(coords_pts[j+1] - coords_pts[j]) for j in range(len(coords_pts)-1)) 
                
                branch_radii = radii[br['pts']]
                r_hyd = float((np.mean(branch_radii**(-4)))**(-0.25)) # Harmonic radius mean for hydraulic resistance

                final_segments[b_id] = { 
                    'n_in': n_in, 'n_out': n_out, 'length': float(length), 
                    'radius': r_hyd, 
                    'is_inlet': n_in in inlet_nodes, 'is_outlet': n_out in outlet_nodes 
                }
                br['oriented'] = True 
                queue.append(n_out) 

    # PHASE 3: JSON ASSEMBLY & RESISTANCE DEBUG
    last_t = int(tau*1000) if tau else 1000
    model_0d = {
        "simulation_parameters": {
            "number_of_cardiac_cycles": 10,
            "number_of_time_pts_per_cardiac_cycle": last_t,
            "time_step_size": 0.001,
            "output_all_cycles": False,
            "density": 0.00106, "viscosity": viscosity_val,
            "model_name": "Multi_Inlet_Model",
            "steady_initial": True,
            "sim_cycle_to_cycle_percent_error": 0.5,
        },
        "boundary_conditions": [], "junctions": [], "vessels": []
    }

    junction_data = defaultdict(lambda: {"in": [], "out": []})
    E = 1e9 
    h = 0.1 

    for b_id, data in final_segments.items(): 
        r_poiseuille = (8.0 * viscosity_val * data['length']) / (np.pi * data['radius']**4)

        vessel = {
            "vessel_id": b_id, "vessel_name": f"branch{b_id}",
            "vessel_length": data['length'], "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": { 
                "R_poiseuille": r_poiseuille, 
                "L": 0,
                "C": (3.0 * data['length'] * np.pi * data['radius']**3) / (2 * E*h),
                "stenosis_coefficient": 0.0
            },
            "boundary_conditions": {}
        }

        if data['is_inlet']: 
            bc_name = f"INLET_{b_id}"
            vessel["boundary_conditions"]["inlet"] = bc_name 
            f_path = flow_files.get(b_id) if flow_files else None 
            model_0d["boundary_conditions"].append({
                "bc_name": bc_name, "bc_type": "FLOW", 
                "bc_values": read_flow_file(f_path) if f_path else {}
            })
        else:
            junction_data[data['n_in']]["out"].append(b_id) 

        if data['is_outlet']: 
            bc_name = f"RCR_{b_id}"
            vessel["boundary_conditions"]["outlet"] = bc_name
            vals = rcr_values.get(b_id, [1000.0, 1e-6, 5000.0]) if rcr_values else [1000.0, 1e-6, 5000.0]
            model_0d["boundary_conditions"].append({
                "bc_name": bc_name, "bc_type": "RCR",
                "bc_values": {"Rp": vals[0], "C": vals[1], "Rd": vals[2], "Pd": 0.0}
            })
        else:
            junction_data[data['n_out']]["in"].append(b_id) 

        model_0d["vessels"].append(vessel)

    for n_id, paths in junction_data.items():
        if paths["in"] or paths["out"]:
            model_0d["junctions"].append({
                "junction_name": f"J{n_id}", "inlet_vessels": paths["in"], 
                "outlet_vessels": paths["out"], "junction_type": "NORMAL_JUNCTION"
            })

    with open(output_path, 'w') as f:
        json.dump(model_0d, f, indent=4)
    print(f"\nCOMPLETED. Initial JSON saved: {output_path}\n" + "="*65)
    return final_segments, pos_to_node, inlet_nodes, outlet_nodes

def visualize_graph(final_segments, pos_to_node, inlet_nodes, outlet_nodes, patient_num, save_path):
    plt.close('all')
    G = nx.DiGraph() 
    node_to_pos = {v: (k[0], k[1]) for k, v in pos_to_node.items()} 

    for b_id, data in final_segments.items():
        G.add_edge(data['n_in'], data['n_out'], id=b_id)

    node_colors = []
    for node in G.nodes():
        if node in inlet_nodes:
            node_colors.append('lightgreen') 
        elif node in outlet_nodes:
            node_colors.append('salmon')      
        else:
            node_colors.append('skyblue')    

    fig, ax = plt.subplots(figsize=(10, 10))
    
    nx.draw_networkx_nodes(G, node_to_pos, node_size=300, node_color=node_colors, 
                           edgecolors='black', ax=ax)
    
    nx.draw_networkx_edges(G, node_to_pos, arrowstyle='->', arrowsize=15, 
                           edge_color='gray', width=1.5, alpha=0.7, ax=ax)
    
    nx.draw_networkx_labels(G, node_to_pos, font_size=8, ax=ax)

    ax.set_title(f"PACS{patient_num:03d} graph", fontweight='bold')

    ax.scatter([], [], c='lightgreen', edgecolors='black', label='Inlet')
    ax.scatter([], [], c='salmon', edgecolors='black', label='Outlet (RCR)')
    ax.scatter([], [], c='skyblue', edgecolors='black', label='Junctions')
    ax.legend(scatterpoints=1, frameon=True, loc='upper right')
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def get_clinical_data(file, p_number):
    pressure_data = pd.read_csv(file)
    row = pressure_data[pressure_data['subject'] == p_number]
    print(row)
    if row.empty:
        raise ValueError(f"Patient {p_number} not found")
    
    sbp = float(row['SBP'].values[0])
    dbp = float(row['DBP'].values[0])
    viscosity = float(row['VISCOSITY'].values[0])
        
    map_pressure = (sbp + 2 * dbp) / 3 
    pulse_pressure = sbp - dbp

    data = row.iloc[0, 5:].to_dict()
    flow_dict = {k.strip(): v*1000.0 for k, v in data.items()}      
    print(f"\nClinical data for PACS{p_number:03d}")
    print(f"Mean pressure: {map_pressure:.2f} mmHg")
    print(f"Pulse: {pulse_pressure:.2f} mmHg")
    print("-" * 40)
    print(f"{'Vessel':<15} | {'Flow (mm3/s)':>15}")
    print("-" * 40)
    for artery, flow in flow_dict.items():
        print(f"{artery:<15} | {flow:>15.2f}")
    print("-" * 40 + "\n")

    clinical_data = {
        "mean_p": map_pressure,
        "pulse": pulse_pressure,
        "viscosity":viscosity,
        "flows": flow_dict,
    }
    
    return clinical_data


def get_initial_BC(clinical_data_file, patient_number, mapping_dict, tau = 1.022):
    my_rcrs = {}
    clinical_data = get_clinical_data(file=clinical_data_file, p_number=patient_number)
    p_mean, p_pulse, viscosity, flow_dict = clinical_data["mean_p"], clinical_data["pulse"],clinical_data["viscosity"], clinical_data["flows"]
    for region_name, branch_id in mapping_dict.items():
        if region_name in flow_dict:
            q_mean = flow_dict[region_name]

            p_dyn = p_mean * 133.3
            r_total = p_dyn / q_mean
        
            r_p = 0.1 * r_total
            r_d = 0.9 * r_total
            
            tau_val = tau  
            c_val = tau_val / r_d
            
            rp = round(r_p, 6)
            rd = round(r_d, 6)
            c = round(c_val, 6)
            
            my_rcrs[branch_id] = [float(rp), float(c), float(rd)]
        else:
            continue
        
    print("\n Initial boundary conditions")
    for bid, values in sorted(my_rcrs.items()):
        print(f"{bid}: {values}")

    return my_rcrs, viscosity

