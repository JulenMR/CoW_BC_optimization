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

def generate_0d_json_multi_inlet(vtp_path, output_path, flow_files=None, rcr_values=None, tau = None):

    if not os.path.exists(vtp_path):
        print(f"Error: File not found at {vtp_path}")
        return

    print(f"\n" + "="*55)
    print(f"POSTPROCESSING CENTERLINE: {os.path.basename(vtp_path)}")
    print("="*55)

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

    raw_branches = {} # Information about the branch
    node_to_branches = defaultdict(list) # Information about the connectivity
    inlet_nodes, outlet_nodes = set(), set()

    for i in range(polydata.GetNumberOfCells()): # Iterate through each branch
        cell = polydata.GetCell(i)
        b_id = int(branch_ids[i])
        pt_ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())] # Get all the points
        n_start, n_end = get_node(pt_ids[0]), get_node(pt_ids[-1]) # Select first and last point from the branch
        raw_branches[b_id] = {'nodes': [n_start, n_end], 'pts': pt_ids, 'oriented': False} 
        node_to_branches[n_start].append(b_id)
        node_to_branches[n_end].append(b_id)
        for p_id in [pt_ids[0], pt_ids[-1]]:
            tag = int(usage_tags[p_id])
            if tag == 1: inlet_nodes.add(get_node(p_id)) # Save inlets
            elif tag == 2: outlet_nodes.add(get_node(p_id)) # Save outlets

    # PHASE 2: ORIENTATION with Breadth First Search algorithm
    final_segments = {}
    queue = deque(list(inlet_nodes)) # The BFS starts with inlets in the queue
    while queue:
        curr_node = queue.popleft() # Gets the first node from the queue
        for b_id in node_to_branches[curr_node]: # Iterates for each branch connected to that node
            br = raw_branches[b_id]
            if not br['oriented']: # If it has not been oriented already
                n_in = curr_node # Takes the current node as inlet
                n_out = br['nodes'][1] if br['nodes'][0] == curr_node else br['nodes'][0] # Takes the other node as outlet
                coords_pts = [np.array(polydata.GetPoint(p)) for p in br['pts']]
                length = sum(np.linalg.norm(coords_pts[j+1] - coords_pts[j]) for j in range(len(coords_pts)-1)) 
                final_segments[b_id] = { # Saves information for each  segment
                    'n_in': n_in, 'n_out': n_out, 'length': float(length), # start/end node, radius, length 
                    'radius': float(radii[br['pts'][0]]),
                    'is_inlet': n_in in inlet_nodes, 'is_outlet': n_out in outlet_nodes # checks if the nodes are inlet or outlet
                }
                br['oriented'] = True # Sets that branch as oriented
                queue.append(n_out) # Adds the outlet node to the queue

    # PHASE 3: JSON ASSEMBLY
    last_t = int(tau*1000)
    model_0d = {
        "simulation_parameters": {
            "number_of_cardiac_cycles": 10,
            "number_of_time_pts_per_cardiac_cycle": last_t,
            "time_step_size": 0.001,
            "output_all_cycles": False,
            "density": 0.00106, "viscosity": 0.004,
            "model_name": "Multi_Inlet_Model",
            "steady_initial": True,
            "sim_cycle_to_cycle_percent_error": 0.5,
        },
        "boundary_conditions": [], "junctions": [], "vessels": []
    }

    junction_data = defaultdict(lambda: {"in": [], "out": []})
    E = 1e9 # Youngs modulus
    h = 0.1 # wall thickness
    for b_id, data in final_segments.items(): # Iterates every branch
        vessel = {
            "vessel_id": b_id, "vessel_name": f"branch{b_id}",
            "vessel_length": data['length'], "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": { # Applies Poiseuilles laws to get R, C, L
                "R_poiseuille": (8.0 * 0.004 * data['length']) / (np.pi * data['radius']**4), 
                "L": 0, #(0.00106 * data['length']) / (np.pi * data['radius']**2),
                "C": (3.0 * data['length'] * np.pi * data['radius']**3) / (2 * E*h),
                "stenosis_coefficient": 0.0
            },
            "boundary_conditions": {}
        }

        if data['is_inlet']: # If the branch has an inlet sets INFLOW
            bc_name = f"INLET_{b_id}"
            vessel["boundary_conditions"]["inlet"] = bc_name # Adds BC name to the specific vessel
            
            f_path = flow_files.get(b_id) if flow_files else None # Gets the flow file from the key (b_id) of the flow dictionary
            model_0d["boundary_conditions"].append({
                "bc_name": bc_name, "bc_type": "FLOW", 
                "bc_values": read_flow_file(f_path)
            })
            print(f"Inlet {b_id}: Assigned flow from {os.path.basename(f_path) if f_path else 'default'}")
        else:
            junction_data[data['n_in']]["out"].append(b_id) # Saves the b_id as an end of its starting node

        if data['is_outlet']: # If the branch has an outlet sets RCR
            bc_name = f"RCR_{b_id}"
            print(f"Adding: {bc_name}")
            vessel["boundary_conditions"]["outlet"] = bc_name
            vals = rcr_values.get(b_id, [1000.0, 1e-6, 5000.0]) if rcr_values else [1000.0, 1e-6, 5000.0]
            model_0d["boundary_conditions"].append({
                "bc_name": bc_name, "bc_type": "RCR",
                "bc_values": {"Rp": vals[0], "C": vals[1], "Rd": vals[2], "Pd": 0.0}
            })
        else:
            junction_data[data['n_out']]["in"].append(b_id) # Saves the b_id as a start of its ending node

        model_0d["vessels"].append(vessel)

    for n_id, paths in junction_data.items():
        if paths["in"] or paths["out"]:
            model_0d["junctions"].append({
                "junction_name": f"J{n_id}", "inlet_vessels": paths["in"], 
                "outlet_vessels": paths["out"], "junction_type": "NORMAL_JUNCTION"
            })

    with open(output_path, 'w') as f:
        json.dump(model_0d, f, indent=4)
    print(f"\nCOMPLETED. Initial JSON saved: {output_path}")
    return final_segments, pos_to_node, inlet_nodes, outlet_nodes

def visualize_graph(final_segments, pos_to_node, inlet_nodes, outlet_nodes):
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

    plt.figure(figsize=(10, 10))
    
    nx.draw_networkx_nodes(G, node_to_pos, node_size=300, node_color=node_colors, edgecolors='black')
    
    nx.draw_networkx_edges(G, node_to_pos, arrowstyle='->', arrowsize=15, 
                           edge_color='gray', width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, node_to_pos, font_size=8)


    plt.title("CoW graph")
    plt.plot([], [], 'o', color='lightgreen', label='Inlet')
    plt.plot([], [], 'o', color='salmon', label='Outlet (RCR)')
    plt.plot([], [], 'o', color='skyblue', label='Junctions')
    plt.legend(scatterpoints=1)
    
    plt.axis('equal') 
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

def get_clinical_data(file, p_number):
    pressure_data = pd.read_csv(file)
    row = pressure_data[pressure_data['subject'] == p_number]
    
    if row.empty:
        raise ValueError(f"Patient {p_number} not found")
    
    sbp = float(row['SBP'].values[0])
    dbp = float(row['DBP'].values[0])
        
    map_pressure = (sbp + 2 * dbp) / 3 
    pulse_pressure = sbp - dbp

    data = row.iloc[0, 4:].to_dict()
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
    
    return map_pressure, pulse_pressure, flow_dict


def get_initial_BC(clinical_data_file, patient_number, mapping_dict, tau = 1.022):
    my_rcrs = {}
    p_mean,_, flow_dict = get_clinical_data(file=clinical_data_file, p_number=patient_number)

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

    return my_rcrs

if __name__ == "__main__":

    patient_number = 5
    model_path = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Models"
    centerlines = os.path.join(model_path, "CENTERLINE", "centerline_final.vtp")
    zeroD_simulation_file = os.path.join(model_path, "zeroD_simulation")
    if not os.path.exists(zeroD_simulation_file):
            os.makedirs(zeroD_simulation_file)
            print(f"Created filepath: {zeroD_simulation_file}")
    output_file = os.path.join(zeroD_simulation_file, "zeroD_script.json")
    clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/corrected_subject_targets.csv"


    carotid_left_flow = os.path.join(zeroD_simulation_file, "LICA_smooth.dat")
    carotid_right_flow = os.path.join(zeroD_simulation_file, "RICA_smooth.dat")
    vertebral_left_flow = os.path.join(zeroD_simulation_file, "LVA_smooth.dat")
    vertebral_right_flow = os.path.join(zeroD_simulation_file, "RVA_smooth.dat")

    flow_data = np.loadtxt(carotid_left_flow)
    tau_param = np.round(flow_data[-1, 0],3)
    print(f"tau: {tau_param}")
    mapping_dict = {
        "SCA_L":4,
        "PCA_L":5,
        "MCA_L":6,
        "ACA_L":7,
        "ACA_R":8,
        "MCA_R":9,
        "PCA_R":10,
        "SCA_R":11,
    }

    my_rcrs = get_initial_BC(clinical_data_file=clinical_data_file, patient_number=patient_number, mapping_dict = mapping_dict, tau=tau_param)
    my_flows = {
        0: carotid_left_flow,
        1: carotid_right_flow,
        2: vertebral_left_flow,
        3: vertebral_right_flow
    }

    segments, pos_to_node, inlet_nodes, outlet_nodes = generate_0d_json_multi_inlet(centerlines, output_file, my_flows, my_rcrs, tau=tau_param)
    #visualize_graph(segments, pos_to_node, inlet_nodes, outlet_nodes)

                          