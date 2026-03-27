import vtk
from vtk.util import numpy_support
import numpy as np
import json
import os
from collections import deque, defaultdict

def generate_0d_json_multi_inlet(vtp_path, output_path, flow_files=None, rcr_values=None):
    """
    flow_files: dict {branch_id: "path/to/file.flow"}
    rcr_values: dict {branch_id: [Rp, C, Rd]}
    """
    if not os.path.exists(vtp_path):
        print(f"Error: File not found at {vtp_path}")
        return

    print(f"\n" + "="*40)
    print(f"PROCESSING MULTI-INLET MODEL: {os.path.basename(vtp_path)}")
    print("="*40)

    def read_flow_file(path):
        if path and os.path.exists(path):
            data = np.loadtxt(path)
            return {"t": data[:, 0].tolist(), "Q": data[:, 1].tolist()}
        return {"t": [0.0, 1.0], "Q": [0.5, 0.5]} # Default fallback

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    # --- PHASE 1: EXTRACTION ---
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
        for p_id in pt_ids:
            tag = int(usage_tags[p_id])
            if tag == 1: inlet_nodes.add(get_node(p_id))
            elif tag == 2: outlet_nodes.add(get_node(p_id))

    # --- PHASE 2: ORIENTATION ---
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
                    'n_in': n_in, 'n_out': n_out, 'length': float(length), 
                    'radius': float(radii[br['pts'][0]]),
                    'is_inlet': n_in in inlet_nodes, 'is_outlet': n_out in outlet_nodes
                }
                br['oriented'] = True
                queue.append(n_out)

    # --- PHASE 3: JSON ASSEMBLY ---
    model_0d = {
        "simulation_parameters": {
            "number_of_cardiac_cycles": 1,
            "number_of_time_pts_per_cardiac_cycle": 200,
            "time_step_size": 0.001,
            "density": 1.06, "viscosity": 0.04,
            "model_name": "Multi_Inlet_Model"
        },
        "boundary_conditions": [], "junctions": [], "vessels": []
    }

    junction_data = defaultdict(lambda: {"in": [], "out": []})

    for b_id, data in final_segments.items():
        vessel = {
            "vessel_id": b_id, "vessel_name": f"branch{b_id}",
            "vessel_length": data['length'], "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "R_poiseuille": (8.0 * 0.04 * data['length']) / (np.pi * data['radius']**4),
                "L": (1.06 * data['length']) / (np.pi * data['radius']**2),
                "C": (3.0 * data['length'] * np.pi * data['radius']**3) / (2.0 * 0.05 * 1e6),
                "stenosis_coefficient": 0.0
            },
            "boundary_conditions": {}
        }

        if data['is_inlet']:
            bc_name = f"INLET_{b_id}"
            vessel["boundary_conditions"]["inlet"] = bc_name
            
            # Look up specific flow file for this branch
            f_path = flow_files.get(b_id) if flow_files else None
            model_0d["boundary_conditions"].append({
                "bc_name": bc_name, "bc_type": "FLOW", 
                "bc_values": read_flow_file(f_path)
            })
            print(f"   Inlet {b_id}: Assigned flow from {os.path.basename(f_path) if f_path else 'default'}")
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

    # (Junction generation)
    for n_id, paths in junction_data.items():
        if paths["in"] or paths["out"]:
            model_0d["junctions"].append({
                "junction_name": f"J{n_id}", "inlet_vessels": paths["in"], 
                "outlet_vessels": paths["out"], "junction_type": "NORMAL_JUNCTION"
            })

    with open(output_path, 'w') as f:
        json.dump(model_0d, f, indent=4)
    print(f"\nCOMPLETED. Multi-inlet JSON saved: {output_path}")

# --- EXAMPLE USAGE ---
my_flows = {
    6: "carotid_left.flow",
    1: "carotid_right.flow",
    17: "vertebral_left.flow",
    11: "vertebral_right.flow"
}

my_rcrs = {
    0: [1000.0, 1e-6, 5000.0],
    15: [1100.0, 1e-6, 5500.0]
    # ... add all 8 outlets
}

# generate_0d_json_multi_inlet(vtp_input, json_output, flow_files=my_flows, rcr_values=my_rcrs)

# --- Ejecución ---
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines"
centerlines = os.path.join(og_dir, "cow_full_final.vtp")
output_file = os.path.join(og_dir, "zeroD_script_full.json")

carotid_left_flow = os.path.join(og_dir, "carotid_left.flow")
carotid_right_flow = os.path.join(og_dir, "carotid_right.flow")
vertebral_left_flow = os.path.join(og_dir, "vertebral_left.flow")
vertebral_right_flow = os.path.join(og_dir, "vertebral_right.flow")

my_flows = {
    6: carotid_left_flow,
    1: carotid_right_flow,
    17: vertebral_right_flow,
    11: vertebral_right_flow
}

my_rcrs = {
    0: [1000.0, 1e-6, 5000.0],
    15: [1100.0, 1e-6, 5500.0],
    16: [1400.0, 1e-6, 5000.0],
    7: [1500.0, 1e-6, 5000.0],
    18: [1500.0, 1e-6, 5000.0],
    2: [1050.0, 1e-6, 5000.0],
    14: [1400.0, 1e-6, 5000.0],
    12: [1700.0, 1e-6, 5000.0]
}

generate_0d_json_multi_inlet(centerlines, output_file, my_flows, my_rcrs)
                          
                          