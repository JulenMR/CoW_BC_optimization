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
from SMC import *
import glob
import time

mapping_dict = {
        "L_ICA":0, "R_ICA":1, "L_VA":2, "R_VA":3, "L_SCA":4, "L_PCA":5,
        "L_MCA":6, "L_ACA":7, "R_ACA":8, "R_MCA":9, "R_PCA":10, "R_SCA":11,
    }

def centerline_extraction(sv_project_filepath, merging_tolerances, extract_individual_centerlines):

    centerlines_file = os.path.join(sv_project_filepath, "ROMSimulations", "Centerlines")
    if not os.path.exists(centerlines_file):
        os.makedirs(centerlines_file)
        print(f"Created filepath: {centerlines_file}")
    
    input_file = os.path.join(sv_project_filepath, "Models", "cow_coarse_model.vtp")
    ModelFaceID_file = os.path.join(sv_project_filepath, "Models", "cow_faceID.mdl")
    final_centerline_file = os.path.join(centerlines_file, "centerline_final.vtp")

    # Face mapping
    df_faceID = pd.read_xml(ModelFaceID_file, xpath=".//face", parser="etree")
    df_caps = df_faceID[df_faceID['type'] == 'cap']
    face_mapping = dict(zip(df_caps['name'], df_caps['id'].astype(int)))

    if extract_individual_centerlines == True:
        extract_individual_paths(input_model_file=input_file, face_mapping=face_mapping, save_file=centerlines_file, custom_objective_branches = None)

    branch_files = glob.glob(os.path.join(centerlines_file, "indbr_*"))

    general_tolerance = merging_tolerances["General tolerance"]
    aca_tolerance = merging_tolerances["ACA tolerance"]
    spatial_tolerance = merging_tolerances["Spatial tolerance"]

    centerline_merging(branch_files=branch_files, input_model_file=input_file, output_file=final_centerline_file,
                        face_mapping = face_mapping, tol_general=general_tolerance, tol_aca = aca_tolerance, spatial_tolerance=spatial_tolerance)
    
def bc_optimization(patient_number, sv_project_filepath, inflows_filepath, clinical_data_csv):

    zeroD_simulation_file = os.path.join(sv_project_filepath, "ROMSimulations", "zeroD_simulation")
    if not os.path.exists(zeroD_simulation_file):
            os.makedirs(zeroD_simulation_file)
            print(f"Created filepath: {zeroD_simulation_file}")    
    
    final_centerline_file = os.path.join(sv_project_filepath, "ROMSimulations", "Centerlines", "centerline_final.vtp")
    initial_json_file = os.path.join(zeroD_simulation_file, "zeroD_script.json")

    # Inflow smoothing 
    opt_3D_simulation_file = os.path.join(sv_project_filepath, "Simulations", "fine", f"{patient_number}_optimized_BC")
    if not os.path.exists(opt_3D_simulation_file):
            os.makedirs(opt_3D_simulation_file)
            print(f"Created filepath: {opt_3D_simulation_file}")   
    smooth_inflow(original_flow_file=inflows_filepath, zeroDsim_file=zeroD_simulation_file, threeDsim_file=opt_3D_simulation_file)

    carotid_left_flow = os.path.join(zeroD_simulation_file, "LICA_0d_smooth.dat")
    carotid_right_flow = os.path.join(zeroD_simulation_file, "RICA_0d_smooth.dat")
    vertebral_left_flow = os.path.join(zeroD_simulation_file, "LVA_0d_smooth.dat")
    vertebral_right_flow = os.path.join(zeroD_simulation_file, "RVA_0d_smooth.dat")

    flow_data = np.loadtxt(carotid_left_flow)
    tau_param = np.round(flow_data[-1, 0],3)

    my_rcrs = get_initial_BC(clinical_data_file=clinical_data_csv, patient_number=patient_number,
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
    clinical_data = get_clinical_data(file=clinical_data_file, p_number=patient_number)
    opt_json = run_optimization(initial_json_file, clinical_data, mapping_dict)
    
    with open(optimized_json, "w") as f:
        json.dump(opt_json, f, indent=4)
    print(f"\nOptimization finalized. JSON File saved in {optimized_json}")

    inp_optimized_path = os.path.join(opt_3D_simulation_file, "svFSI_optimized.inp")
    update_svfsi(json_path=optimized_json, inp_path="svFSI_base.inp", save_path = inp_optimized_path, mapping_dict=mapping_dict)

def uncertainty_quantification(sv_project_filepath, patient_number, clinical_data_file, num_particles, num_cores, error_tolerance):

    clinical_data  = get_clinical_data(file=clinical_data_file, p_number=patient_number)
    json_path = os.path.join(sv_project_filepath, "ROMSimulations", "zeroD_simulation", "zeroD_script_optimized.json")

    with open(json_path, 'r') as f: json_dict = json.load(f)
    active_rcr_ids = []
    deterministic_param_values = []
    for i in range(4, 12):
        bc_list = json_dict["boundary_conditions"]
        try:
            params = next(bc["bc_values"] for bc in bc_list if bc.get("bc_name") == f"RCR_{i}")
            for n, val in enumerate(params.values()):
                if n < 3:
                  deterministic_param_values.append(round(val,6))
            active_rcr_ids.append(i)
        except StopIteration:
            continue
    
    rcr_model = RCR_UQ(json_dict=json_dict, 
                       active_rcr_ids=active_rcr_ids, 
                       clinical_targets=clinical_data, 
                       branch_map=mapping_dict, 
                       lbfgs_vals = deterministic_param_values,
                       error_tolerance= error_tolerance)

    fk_boot = ssm.Bootstrap(ssm=rcr_model, data=np.zeros(1))

    print(f"Quantifying Uncertainty executing Sequential Monte Carlo with {num_particles} particles in {num_cores} cores")

    results = particles.multiSMC(fk=fk_boot, 
                                 N=num_particles, 
                                 nruns=1, 
                                 nprocs=num_cores, 
                                 out_func=None)

    alg = results[0]['output']

    smc_results_files = "SMC_results"
    if not os.path.exists(smc_results_files):
            os.makedirs(smc_results_files)
    np.save(os.path.join(smc_results_files, f"smc_result_pacs{patient_number:03d}.npy"), alg.X)
    
    print(f"SMC was successfull. {alg.X.shape[0]} samples for {alg.X.shape[1]} parameters have been created.")

if __name__ == "__main__":
    patient_number = 8
    sv_path = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}"
    merging_tolerances = {
         "General tolerance": 0.01,
         "ACA tolerance": 0.01,
         "Spatial tolerance": 2.2
    }

    inflow_files = os.path.join(sv_path, "Simulations", "fine", f"{patient_number}_asl")
    clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/subject_targets.csv"

    num_particles = 200
    n_cores = 16
    err_tolerance = 0.05

    centerline_extraction(sv_project_filepath = sv_path, merging_tolerances = merging_tolerances, extract_individual_centerlines = False)

    bc_optimization(patient_number = patient_number, sv_project_filepath = sv_path, inflows_filepath = inflow_files, clinical_data_csv = clinical_data_file)

    uncertainty_quantification(sv_project_filepath = sv_path, patient_number = patient_number, clinical_data_file = clinical_data_file, 
                               num_particles = num_particles, num_cores = n_cores, error_tolerance = err_tolerance)


