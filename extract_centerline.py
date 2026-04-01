import vtk
from vmtk import pypes
import os
import glob
import pandas as pd
import numpy as np

def get_face_center(polydata, array_name, face_id):
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
    """Crea un objeto nuevo dividiendo la línea en segmentos para asegurar conectividad."""
    old_points = polydata.GetPoints()
    num_pts = old_points.GetNumberOfPoints()
    
    if num_pts < 2: return polydata
    
    new_points = vtk.vtkPoints()
    for i in range(num_pts):
        new_points.InsertNextPoint(old_points.GetPoint(i))
    
    # --- CAMBIO AQUÍ: Crear segmentos individuales (0-1, 1-2, 2-3...) ---
    new_lines = vtk.vtkCellArray()
    for i in range(num_pts - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        new_lines.InsertNextCell(line)
    
    new_poly = vtk.vtkPolyData()
    new_poly.SetPoints(new_points)
    new_poly.SetLines(new_lines)
    
    # Copiar radios si existen
    old_radii = polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if old_radii:
        new_poly.GetPointData().AddArray(old_radii)
        
    return new_poly

# SETTINGS
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models"
save_file = os.path.join(og_dir, "FULL_PIPELINE")
input_file = os.path.join(og_dir, "no_collaterals.vtp")
xml_file = os.path.join(og_dir, "no_collaterals.mdl")
cell_data_array = "ModelFaceID" 
scaling_factor = 0.1
scaled_file = os.path.join(save_file, "scaled.vtp")

# FaceID mapping
df_faceID = pd.read_xml(xml_file, xpath=".//face")

face_mapping = dict(zip(df_faceID['name'],df_faceID['id']))
for k, v in face_mapping.items():
    print(f"k: {k}, v: {v}")

inlet_names = ["cap_LeftVert_Basilar_LeftPost", "cap_RightVert", "cap_Right_SCA", "cap_Right_Post",
                "cap_Left_ICA_MCA_2", "cap_Left_ICA_MCA", "cap_Right_ICA_MCA"]

outlet_names = ["cap_Right_Post", "cap_Left_SCA", "cap_LeftVert_Basilar_LeftPost_2", "cap_Right_ICA_MCA_2",
                "cap_Right_Anterior", "cap_Left_ICA_MCA_2", "cap_Left_Anterior"]

in_ids = []
out_ids = []

for cap in inlet_names:
    in_ids.append(face_mapping[cap])

for cap in outlet_names:
    out_ids.append(face_mapping[cap])

# Cargar la malla original una sola vez
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(input_file)
reader.Update()
mesh = reader.GetOutput()

transform = vtk.vtkTransform()
transform.Scale(scaling_factor, scaling_factor, scaling_factor)

transformFilter = vtk.vtkTransformPolyDataFilter()
transformFilter.SetInputData(mesh)
transformFilter.SetTransform(transform)
transformFilter.Update()

mesh = transformFilter.GetOutput()
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(scaled_file)
writer.SetInputData(mesh)
writer.Write()

print(f"[RECONSTRUIDO] Objeto creado desde cero en {scaled_file}")

print("--- FASE 1: Extracción de segmentos individuales con integridad topográfica ---")
for i in range(len(in_ids)):
    in_id = in_ids[i]
    out_id = out_ids[i]
    
    in_point = get_face_center(mesh, cell_data_array, in_id)
    out_point = get_face_center(mesh, cell_data_array, out_id)
    
    if in_point is None or out_point is None:
        continue

    temp_out = os.path.join(save_file, f"tmp_cl_{in_id}_{out_id}.vtp")
    fixed_temp_out = os.path.join(save_file, f"fixed_tmp_cl_{in_id}_{out_id}.vtp")
    
    # NUEVO COMANDO BASADO EN LA DOCUMENTACIÓN:
    # Usamos -resampling 1 para activar el remuestreo interno
    # Usamos -resamplingstep 0.1 para definir la distancia entre puntos
    vmtk_cmd = (
        f'vmtksurfacereader -ifile {scaled_file} '
        f'--pipe vmtkcenterlines -seedselector pointlist '
        f'-sourcepoints {in_point[0]} {in_point[1]} {in_point[2]} '
        f'-targetpoints {out_point[0]} {out_point[1]} {out_point[2]} '
        f'-capdisplacement 0.0 '  # Evita que VMTK empuje el punto hacia adentro
        f'-endpoints 1 '          # Fuerza a incluir los baricentros de las tapas
        f'-resampling 1 -resamplingstep 0.1 ' # Mantén el paso fino
    )
    
    print(f"\n>> Procesando camino: {in_id} a {out_id}")
    myPype = pypes.PypeRun(vmtk_cmd)

    cl_obj = myPype.GetScriptObject('vmtkcenterlines', '0').Centerlines

    if cl_obj:
        # RECONSTRUCCIÓN TOTAL
        fixed_cl = rebuild_polydata_from_scratch(cl_obj)
        
        # GUARDADO
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(fixed_temp_out)
        writer.SetInputData(fixed_cl)
        writer.Write()
        print(f"[RECONSTRUIDO] Objeto creado desde cero en {fixed_temp_out}")


