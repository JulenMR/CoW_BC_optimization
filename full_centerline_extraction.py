import vtk
from vmtk import pypes
import os
import pandas as pd
import numpy as np
from individual_centerline import *
from merge_centerlines import *
import glob

"""
# 326_no_collaterals
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models"
save_file = os.path.join(og_dir, "FULL_PIPELINE")
input_file = os.path.join(og_dir, "no_collaterals.vtp")
xml_file = os.path.join(og_dir, "no_collaterals.mdl")

inlet_names = ["cap_LeftVert_Basilar_LeftPost", "cap_RightVert", "cap_Right_SCA", "cap_Right_Post",
                "cap_Left_ICA_MCA_2", "cap_Left_ICA_MCA", "cap_Right_ICA_MCA"]
outlet_names = ["cap_Right_Post", "cap_Left_SCA", "cap_LeftVert_Basilar_LeftPost_2", "cap_Right_ICA_MCA_2",
                "cap_Right_Anterior", "cap_Left_ICA_MCA_2", "cap_Left_Anterior"]
MASTER_INFLOWS = ["cap_Left_ICA_MCA", "cap_Right_ICA_MCA", "cap_LeftVert_Basilar_LeftPost", "cap_RightVert"]
MASTER_OUTFLOWS = ["cap_Left_SCA", "cap_LeftVert_Basilar_LeftPost_2", "cap_Left_ICA_MCA_2", "cap_Left_Anterior", "cap_Right_Anterior", "cap_Right_ICA_MCA_2", "cap_Right_Post", "cap_Right_SCA"]

"""

if __name__ == "__main__":
    
    og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-005/Models"

    save_file = os.path.join(og_dir, "FULL_PIPELINE_FAST")
    input_file = os.path.join(og_dir, "cow_super_coarse.vtp")
    xml_file = os.path.join(og_dir, "cow.mdl")
    scaled_filename = os.path.join(save_file, "scaled_model.vtp")

    final_centerline_file = os.path.join(save_file, "final_centerline.vtp")

    # Face mapping
    df_faceID = pd.read_xml(xml_file, xpath=".//face", parser="etree")
    df_caps = df_faceID[df_faceID['type'] == 'cap']
    face_mapping = dict(zip(df_caps['name'], df_caps['id'].astype(int)))

    """
    extract_individual_paths(input_model_file=input_file, face_mapping=face_mapping, save_file=save_file,
                              scaled_filename=scaled_filename)
    """
    branch_files = glob.glob(os.path.join(save_file, "individual_branch_*"))
    centerline_merging(branch_files=branch_files, scaled_model_path=scaled_filename, output_file=final_centerline_file,
                        face_mapping = face_mapping, tolerance_cleaning=0.011, spatial_tolerance=2)

