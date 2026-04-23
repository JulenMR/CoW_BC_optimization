import vtk
from vmtk import pypes
import os
import pandas as pd
import numpy as np
from individual_centerline import *
from merge_centerlines import *
from generate_json import *
from rcr_optimization import *
from create_3dsim_file import *
from inflow_smoothing import *
import glob
import time

if __name__ == "__main__":
    patient_number = 8
    start_time = time.time()
    model_path = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Models"
    threeD_simulation_path = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Simulations/fine"

    centerlines_file = os.path.join(model_path, "CENTERLINES")
    if not os.path.exists(centerlines_file):
        os.makedirs(centerlines_file)
        print(f"Created filepath: {centerlines_file}")
    
    input_file = os.path.join(model_path, "cow_super_coarse.vtp")
    ModelFaceID_file = os.path.join(model_path, "cow.mdl")
    final_centerline_file = os.path.join(centerlines_file, "centerline_final.vtp")

    # Face mapping
    df_faceID = pd.read_xml(ModelFaceID_file, xpath=".//face", parser="etree")
    df_caps = df_faceID[df_faceID['type'] == 'cap']
    face_mapping = dict(zip(df_caps['name'], df_caps['id'].astype(int)))


    # PHASE 1: Centerline extraction
    #extract_individual_paths(input_model_file=input_file, face_mapping=face_mapping, save_file=centerlines_file, custom_objective_branches = None)
    
    branch_files = glob.glob(os.path.join(centerlines_file, "indbr_*"))
    aca_files = [f for f in branch_files if "ACA" in os.path.basename(f)]
    rest_files = [f for f in branch_files if "ACA" not in os.path.basename(f)]

    centerline_merging(branch_files=branch_files, input_model_file=input_file, output_file=final_centerline_file,
                        face_mapping = face_mapping, tol_high=0.01, tol_low = 0.01, spatial_tolerance=2.5)
    
    # Phase 2: Generate JSON file
    zeroD_simulation_file = os.path.join(model_path, "zeroD_simulation")
    if not os.path.exists(zeroD_simulation_file):
            os.makedirs(zeroD_simulation_file)
            print(f"Created filepath: {zeroD_simulation_file}")    

    initial_json_file = os.path.join(zeroD_simulation_file, "zeroD_script.json")
    clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/subject_targets.csv"

    # Inflow smoothing 
    original_flow_file = os.path.join(threeD_simulation_path, "8_asl")
    opt_3D_simulation_file = os.path.join(threeD_simulation_path, f"{patient_number}_optimized_BC")
    if not os.path.exists(opt_3D_simulation_file):
            os.makedirs(opt_3D_simulation_file)
            print(f"Created filepath: {opt_3D_simulation_file}")   
    smooth_inflow(original_flow_file=original_flow_file, zeroDsim_file=zeroD_simulation_file, threeDsim_file=opt_3D_simulation_file)

    carotid_left_flow = os.path.join(zeroD_simulation_file, "LICA_0d_smooth.dat")
    carotid_right_flow = os.path.join(zeroD_simulation_file, "RICA_0d_smooth.dat")
    vertebral_left_flow = os.path.join(zeroD_simulation_file, "LVA_0d_smooth.dat")
    vertebral_right_flow = os.path.join(zeroD_simulation_file, "RVA_0d_smooth.dat")

    flow_data = np.loadtxt(carotid_left_flow)
    tau_param = np.round(flow_data[-1, 0],3)

    mapping_dict = {
        "L_ICA":0, "R_ICA":1, "L_VA":2, "R_VA":3, "L_SCA":4, "L_PCA":5,
        "L_MCA":6, "L_ACA":7, "R_ACA":8, "R_MCA":9, "R_PCA":10, "R_SCA":11,
    }

    my_rcrs = get_initial_BC(clinical_data_file=clinical_data_file, patient_number=patient_number,
                             mapping_dict=mapping_dict, tau=tau_param)
               
    my_flows = {
        0: carotid_left_flow,
        1: carotid_right_flow,
        2: vertebral_left_flow,
        3: vertebral_right_flow
    }

    segments, pos_to_node, inlet_nodes, outlet_nodes = generate_0d_json_multi_inlet(vtp_path= final_centerline_file, output_path = initial_json_file, 
                                                                                    flow_files=my_flows, rcr_values=my_rcrs, tau=tau_param)

    optimized_json = os.path.join(zeroD_simulation_file, "zeroD_script_optimized.json")
    mean_p, pulse, clinical_flows  = get_clinical_data(file=clinical_data_file, p_number=patient_number)
    iteration_count = 0

    # opt_json = run_optimization(initial_json_file, mean_p, pulse, clinical_flows, mapping_dict)
    
    # with open(optimized_json, "w") as f:
    #     json.dump(opt_json, f, indent=4)
    # print(f"\nOptimization finalized. JSON File saved in {optimized_json}")

    inp_path = os.path.join(original_flow_file, "svFSI.inp")
    inp_optimized_path = os.path.join(opt_3D_simulation_file, "svFSI_optimized.inp")
    update_svfsi(json_path=optimized_json, inp_path=inp_path, save_path = inp_optimized_path, mapping_dict=mapping_dict)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

