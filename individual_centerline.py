import vtk
from vmtk import pypes
import os
import pandas as pd
import numpy as np

def get_face_center(polydata, array_name, face_id):
    # Extracts the centerpoint from a face
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, array_name)
    threshold.SetLowerThreshold(face_id)
    threshold.SetUpperThreshold(face_id)
    threshold.Update()
    
    if threshold.GetOutput().GetNumberOfCells() == 0:
        return None
        
    center_filter = vtk.vtkCenterOfMass()
    center_filter.SetInputData(threshold.GetOutput())
    center_filter.SetUseScalarsAsWeights(False)
    center_filter.Update()
    return center_filter.GetCenter()

def rebuild_polydata_from_scratch(polydata):
    old_points = polydata.GetPoints()
    num_pts = old_points.GetNumberOfPoints()
    
    if num_pts < 2: return polydata
    
    new_points = vtk.vtkPoints()
    for i in range(num_pts):
        new_points.InsertNextPoint(old_points.GetPoint(i))
    
    new_lines = vtk.vtkCellArray()
    for i in range(num_pts - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        new_lines.InsertNextCell(line)
    
    new_poly = vtk.vtkPolyData()
    new_poly.SetPoints(new_points)
    new_poly.SetLines(new_lines)
    
    old_radii = polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if old_radii:
        new_poly.GetPointData().AddArray(old_radii)
        
    return new_poly


def extract_individual_paths(input_model_file, face_mapping, save_file, custom_objective_branches=None):


    if custom_objective_branches is not None:
        objective_branches = custom_objective_branches
    else:
        objective_branches = [
        ("cap_L_ICA", "cap_L_ICA_2"),
        ("cap_R_ICA", "cap_R_ICA_2"),
        ("cap_L_SCA", "cap_R_ACA"),
        ("cap_R_SCA", "cap_L_ACA"),
        ("cap_L_VA", "cap_R_PCA"),
        ("cap_R_VA", "cap_R_VA_2"),
        ("cap_L_ICA_2", "cap_R_ICA_2")
    ]
    all_cap_names = ["cap_L_ICA", "cap_L_ICA_2", "cap_R_ICA", "cap_R_ICA_2", "cap_L_SCA", "cap_R_ACA", "cap_R_SCA", "cap_L_ACA",
                     "cap_L_VA", "cap_R_PCA", "cap_R_VA", "cap_R_VA_2"]
    currents_caps = list(face_mapping.keys())
    missing_caps = list(set(all_cap_names) - set(currents_caps))
    subsitution_pairs = {
        "cap_L_ICA":"cap_R_VA_2",
        "cap_L_ICA_2":"cap_L_ACA",
        "cap_R_ICA":"cap_R_PCA",
        "cap_R_ICA_2":"cap_R_ACA",
        "cap_L_VA":"cap_R_VA",
        "cap_L_SCA":"cap_R_SCA"
    }

    if len(missing_caps)>0:
        fixed_point = "cap_L_ICA" if "cap_L_ICA" in currents_caps else currents_caps[0]
        objective_branches = [
        (
            subsitution_pairs.get(start, fixed_point) if start in missing_caps else start,
            subsitution_pairs.get(end, fixed_point) if end in missing_caps else end
        )
        for start, end in objective_branches
    ]
        

    # Load original mesh
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_model_file)
    reader.Update()
    mesh = reader.GetOutput()


    print("Phase 1: Extraction of individual branches")
    for start_name, end_name in objective_branches:
        if start_name not in face_mapping or end_name not in face_mapping:
            print(f"  Skipping {start_name} -> {end_name}: Cap not found")
            continue
            
        in_id = face_mapping[start_name]
        out_id = face_mapping[end_name]
        
        in_point = get_face_center(mesh, "ModelFaceID", in_id)
        out_point = get_face_center(mesh, "ModelFaceID", out_id)
        
        if in_point is None or out_point is None:
            continue

        fixed_temp_out = os.path.join(save_file, f"individual_branch_{in_id}_{out_id}.vtp")

        vmtk_cmd = (
            f'vmtksurfacereader -ifile {input_model_file} '
            f'--pipe vmtkcenterlines -seedselector pointlist '
            f'-sourcepoints {in_point[0]} {in_point[1]} {in_point[2]} '
            f'-targetpoints {out_point[0]} {out_point[1]} {out_point[2]} '
            f'-capdisplacement 0.0 '  
            f'-endpoints 1 '          
            f'-resampling 1 -resamplingstep 0.1 ' 
        )
        
        print(f"\n>> Processing path from {in_id} to {out_id}")
        myPype = pypes.PypeRun(vmtk_cmd)

        cl_obj = myPype.GetScriptObject('vmtkcenterlines', '0').Centerlines

        if cl_obj:
            fixed_cl = rebuild_polydata_from_scratch(cl_obj)
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(fixed_temp_out)
            writer.SetInputData(fixed_cl)
            writer.Write()
            print(f"Path rebuilt in {fixed_temp_out}")

    


