import vtk
from vmtk import pypes
import os
import pandas as pd
import numpy as np
from individual_centerline import *
from merge_centerlines import *
from generate_json import *
from rcr_optimization import *
import glob
import time

if __name__ == "__main__":
    patient_number = 11
    start_time = time.time()
    og_dir = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Models"

    save_file = os.path.join(og_dir, "CENTERLINE")
    if not os.path.exists(save_file):
        os.makedirs(save_file)
        print(f"Created filepath: {save_file}")
    
    input_file = os.path.join(og_dir, "cow_super_coarse.vtp")
    ModelFaceID_file = os.path.join(og_dir, "cow.mdl")
    final_centerline_file = os.path.join(save_file, "centerline_final.vtp")

    # Face mapping
    df_faceID = pd.read_xml(ModelFaceID_file, xpath=".//face", parser="etree")
    df_caps = df_faceID[df_faceID['type'] == 'cap']
    face_mapping = dict(zip(df_caps['name'], df_caps['id'].astype(int)))

    # Objective branch pairs
    objective_branches = [
        ("cap_L_ICA", "cap_L_ICA_2"),
        ("cap_R_ICA", "cap_R_ICA_2"),
        ("cap_L_SCA", "cap_L_ICA_2"),
        ("cap_R_SCA", "cap_R_ICA_2"),
        ("cap_L_VA", "cap_R_PCA"),
        ("cap_R_VA", "cap_L_PCA"),
        ("cap_L_ICA_2", "cap_R_ICA_2"),
        ("cap_L_ACA", "cap_R_ACA")
    ]

    # PHASE 1: Centerline extraction

    #extract_individual_paths(input_model_file=input_file, face_mapping=face_mapping, save_file=save_file, custom_objective_branches = objective_branches)
    
    branch_files = glob.glob(os.path.join(save_file, "indbr_*"))
    aca_files = [f for f in branch_files if "ACA" in os.path.basename(f)]
    rest_files = [f for f in branch_files if "ACA" not in os.path.basename(f)]

    centerline_merging(branch_files=branch_files, input_model_file=input_file, output_file=final_centerline_file,
                        face_mapping = face_mapping, tol_high=0.018, tol_low = 0.004, spatial_tolerance=2)
    
    # Phase 2: Generate JSON file
    simulation_file = os.path.join(og_dir, "zeroD_simulation")
    if not os.path.exists(simulation_file):
            os.makedirs(simulation_file)
            print(f"Created filepath: {simulation_file}")    

    initial_json_file = os.path.join(simulation_file, "zeroD_script.json")
    clinical_flows_file = os.path.join(og_dir, f"ASL_BC_subject{patient_number}_FINAL.csv")
    clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/corrected_subject_targets.csv"

    carotid_left_flow = os.path.join(og_dir, "LICA_smooth.dat")
    carotid_right_flow = os.path.join(og_dir, "RICA_smooth.dat")
    vertebral_left_flow = os.path.join(og_dir, "LVA_smooth.dat")
    vertebral_right_flow = os.path.join(og_dir, "RVA_smooth.dat")

    flow_data = np.loadtxt(carotid_left_flow)
    tau_param = np.round(flow_data[-1, 0],3)
    print(f"tau: {tau_param}")
    mapping_dict = {
        "ICA_L":0,
        "ICA_R":1,
        "VA_L":2,
        "VA_R":3,
        "SCA_L":4,
        "PCA_L":5,
        "MCA_L":6,
        "ACA_L":7,
        "ACA_R":8,
        "MCA_R":9,
        "PCA_R":10,
        "SCA_R":11,
    }

    my_rcrs = get_initial_BC(clinical_data_file=clinical_data_file, patient_number=patient_number,
                             mapping_dict=mapping_dict, tau=tau_param)
               
    my_flows = {
        0: carotid_left_flow,
        1: carotid_right_flow,
        2: vertebral_left_flow,
        3: vertebral_right_flow
    }

    segments, pos_to_node, inlet_nodes, outlet_nodes = generate_0d_json_multi_inlet(centerlines, initial_json_file, my_flows, my_rcrs, tau=tau_param)

    clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/corrected_subject_targets.csv"
    optimized_json = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Models/zeroD_simulation/zeroD_script_optimized.json"
    mean_p, pulse, clinical_flows  = get_clinical_data(file=clinical_data_file, p_number=patient_number)
    iteration_count = 0

    BRANCH_MAPPING = {
    "branch0": "ICA_L", "branch1": "ICA_R", "branch2": "VA_L", "branch3": "VA_R", "branch4": "SCA_L", "branch5": "PCA_L", 
    "branch6": "MCA_L", "branch7": "ACA_L", "branch8": "ACA_R", "branch9": "MCA_R", "branch10": "PCA_R", "branch11": "SCA_R"
    }

    opt_json = run_optimization(initial_json_file, mean_p, pulse, clinical_flows, BRANCH_MAPPING)
    
    with open(optimized_json, "w") as f:
        json.dump(opt_json, f, indent=4)
    print(f"\nOptimization finalized. JSON File saved in {optimized_json}")

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

