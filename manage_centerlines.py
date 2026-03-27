import vtk
from vtk.util import numpy_support
import numpy as np
import json
import os
from collections import deque, defaultdict


def generate_0d_json_fixed(vtp_path, output_path):
    if not os.path.exists(vtp_path):
        print(f"Error: File not found at {vtp_path}")
        return

    print(f"\n" + "="*40)
    print(f"PROCESSING: {os.path.basename(vtp_path)}")
    print("="*40)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    # --- PHASE 1: DATA EXTRACTION AND MAPPING ---
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
    inlet_nodes = set()
    outlet_nodes = set()

    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        b_id = int(branch_ids[i])
        pt_ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
        
        n_start, n_end = get_node(pt_ids[0]), get_node(pt_ids[-1])
        raw_branches[b_id] = {'nodes': [n_start, n_end], 'pts': pt_ids, 'oriented': False}
        node_to_branches[n_start].append(b_id)
        node_to_branches[n_end].append(b_id)
        
        for p_id in pt_ids:
            tag = int(usage_tags[p_id])
            if tag == 1: inlet_nodes.add(get_node(p_id))
            elif tag == 2: outlet_nodes.add(get_node(p_id))

    print(f"Initial Scan Summary:")
    print(f"   - Total branches found: {len(raw_branches)}")
    print(f"   - Inlet Nodes detected: {list(inlet_nodes)}")
    print(f"   - Outlet Nodes detected: {list(outlet_nodes)}")

    # --- PHASE 2: ORIENTATION (BFS FOR LOOPS) ---
    print(f"\nStarting flow propagation...")
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

                final_segments[b_id] = {
                    'n_in': n_in, 'n_out': n_out, 
                    'length': float(length), 
                    'radius': float(radii[br['pts'][0]]),
                    'is_inlet': n_in in inlet_nodes,
                    'is_outlet': n_out in outlet_nodes
                }
                br['oriented'] = True
                print(f"   Branch {b_id:2}: Node {n_in:2} -> {n_out:2} oriented.")
                queue.append(n_out)

    # --- PHASE 3: NETWORK ASSEMBLY ---
    rho, mu = 1.06, 0.04
    model_0d = {
        "simulation_parameters": {
            "number_of_cardiac_cycles": 1,
            "number_of_time_pts_per_cardiac_cycle": 100,
            "time_step_size": 0.01,
            "density": rho, "viscosity": mu,
            "model_name": "CoW_Model_Fixed"
        },
        "boundary_conditions": [], "junctions": [], "vessels": []
    }

    junction_data = defaultdict(lambda: {"in": [], "out": []})

    print(f"\nAssembling vessels and boundary conditions...")
    for b_id, data in final_segments.items():
        L, r = data['length'], data['radius']
        R_p = (8.0 * mu * L) / (np.pi * r**4)
        Ind = (rho * L) / (np.pi * r**2)
        Cap = (3.0 * L * np.pi * r**3) / (2.0 * 0.05 * 1e6)

        vessel = {
            "vessel_id": int(b_id),
            "vessel_name": f"branch{b_id}",
            "vessel_length": L,
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {"R_poiseuille": R_p, "L": Ind, "C": Cap, "stenosis_coefficient": 0.0},
            "boundary_conditions": {}
        }

        if data['is_inlet']:
            bc_name = f"INFLOW_{b_id}"
            vessel["boundary_conditions"]["inlet"] = bc_name
            model_0d["boundary_conditions"].append({
                "bc_name": bc_name, "bc_type": "FLOW", "bc_values": {"t": [0.0, 1.0], "Q": [0.5, 0.5]}
            })
            print(f"   Inlet Boundary Condition: Branch {b_id} -> {bc_name}")
        else:
            junction_data[data['n_in']]["out"].append(int(b_id))

        if data['is_outlet']:
            bc_name = f"RESISTANCE_{b_id}"
            vessel["boundary_conditions"]["outlet"] = bc_name
            model_0d["boundary_conditions"].append({
                "bc_name": bc_name, "bc_type": "RESISTANCE", "bc_values": {"R": 1000.0, "Pd": 0.0}
            })
            print(f"   Outlet Boundary Condition: Branch {b_id} -> {bc_name}")
        else:
            junction_data[data['n_out']]["in"].append(int(b_id))

        model_0d["vessels"].append(vessel)

    print(f"\nGenerating Junctions...")
    for n_id, paths in junction_data.items():
        if paths["in"] or paths["out"]:
            model_0d["junctions"].append({
                "junction_name": f"J{n_id}",
                "inlet_vessels": paths["in"],
                "outlet_vessels": paths["out"],
                "junction_type": "NORMAL_JUNCTION"
            })
            print(f"   J{n_id:2}: Inbound {paths['in']} -> Outbound {paths['out']}")

    with open(output_path, 'w') as f:
        json.dump(model_0d, f, indent=4)
    print(f"\nCOMPLETED. JSON saved to: {output_path}\n")


# Ejemplo de uso
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines"
# Procesar ambos ejemplos
generate_0d_json_fixed(os.path.join(og_dir, "easy_example", "cow_full_final.vtp"), 
                          os.path.join(og_dir, "easy_example", "zeroD_script.json"))

generate_0d_json_fixed(os.path.join(og_dir, "medium_example", "cow_medium_final.vtp"), 
                          os.path.join(og_dir, "medium_example", "zeroD_script.json"))

                          
vtp_input = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines/cow_full_final.vtp"
json_output = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines/zeroD_loop_fix.json"
generate_0d_json_fixed(vtp_input, json_output)