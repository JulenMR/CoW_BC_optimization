import vtk
from vmtk import pypes
import os
import pandas as pd
import numpy as np
from individual_centerline import *
from merge_centerlines import *
import glob
import time

if __name__ == "__main__":
    
    start_time = time.time()
    og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-011/Models"

    save_file = os.path.join(og_dir, "CENTERLINE")
    if not os.path.exists(save_file):
        os.makedirs(save_file)
        print(f"Created filepath: {save_file}")
    
    input_file = os.path.join(og_dir, "cow_super_coarse.vtp")
    xml_file = os.path.join(og_dir, "cow.mdl")
    final_centerline_file = os.path.join(save_file, "centerline prueba1.vtp")

    # Face mapping
    df_faceID = pd.read_xml(xml_file, xpath=".//face", parser="etree")
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

    #extract_individual_paths(input_model_file=input_file, face_mapping=face_mapping, save_file=save_file, custom_objective_branches = objective_branches)
    
    branch_files = glob.glob(os.path.join(save_file, "indbr_*"))
    aca_files = [f for f in branch_files if "ACA" in os.path.basename(f)]
    rest_files = [f for f in branch_files if "ACA" not in os.path.basename(f)]

    centerline_merging(branch_files=branch_files, input_model_file=input_file, output_file=final_centerline_file,
                        face_mapping = face_mapping, tol_high=0.004, tol_low = 0.004, spatial_tolerance=3)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

