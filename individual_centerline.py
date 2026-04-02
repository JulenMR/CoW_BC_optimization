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


def extract_individual_paths(input_model_file, face_mapping, save_file, scaled_filename, scaling_factor = 1):

    if not os.path.exists(save_file):
        os.makedirs(save_file)
        print(f"Created filepath: {save_file}")

    inlet_names = ["cap_L_VA", "cap_R_VA", "cap_R_SCA", "cap_R_PCA", "cap_R_VA_2", "cap_L_ICA", "cap_R_ICA"]
    outlet_names = ["cap_R_PCA", "cap_L_SCA", "cap_R_VA_2", "cap_R_ICA_2", "cap_R_ACA", "cap_L_ICA_2", "cap_L_ACA"]

    cell_data_array = "ModelFaceID" 

    in_ids = []
    out_ids = []

    for cap in inlet_names:
        in_ids.append(face_mapping[cap])

    for cap in outlet_names:
        out_ids.append(face_mapping[cap])

    # Load original mesh
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_model_file)
    reader.Update()
    mesh = reader.GetOutput()

    # Scaling
    transform = vtk.vtkTransform()
    transform.Scale(scaling_factor, scaling_factor, scaling_factor)
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(mesh)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    mesh = transformFilter.GetOutput()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(scaled_filename)
    writer.SetInputData(mesh)
    writer.Write()
    print(f"Scaled object saved in {scaled_filename}")

    print("Phase 1: Extraction of individual branches")
    for i in range(len(in_ids)):
        in_id = in_ids[i]
        out_id = out_ids[i]
        
        in_point = get_face_center(mesh, cell_data_array, in_id)
        out_point = get_face_center(mesh, cell_data_array, out_id)
        
        if in_point is None or out_point is None:
            continue

        fixed_temp_out = os.path.join(save_file, f"individual_branch_{in_id}_{out_id}.vtp")

        vmtk_cmd = (
            f'vmtksurfacereader -ifile {scaled_filename} '
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

    


