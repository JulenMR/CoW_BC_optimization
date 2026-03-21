import vtk
from vmtk import pypes
import os
import glob

def get_face_center(polydata, array_name, face_id):
    """Filtra la malla y devuelve las coordenadas (x,y,z) del centro de la cara."""
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

# --- CONFIGURACIÓN ---
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models"
save_file = os.path.join(og_dir, "Centerlines")
input_file = os.path.join(og_dir, "no_collaterals_reindexed.vtp")
cell_data_array = "ModelFaceID" 

# IDs de SimVascular (Inlets y Outlets correspondientes)
#inlet_ids = [12, 16, 20, 19, 22] 
#outlet_ids = [19, 15, 21, 23, 17]
inlet_ids = [14] 
outlet_ids = [22]

# Cargar la malla original una sola vez
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(input_file)
reader.Update()
mesh = reader.GetOutput()

print("--- FASE 1: Extracción de segmentos individuales con integridad topográfica ---")
for i in range(len(inlet_ids)):
    in_id = inlet_ids[i]
    out_id = outlet_ids[i]
    
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
        f'vmtksurfacereader -ifile {input_file} '
        f'--pipe vmtkcenterlines -seedselector pointlist '
        f'-sourcepoints {in_point[0]} {in_point[1]} {in_point[2]} '
        f'-targetpoints {out_point[0]} {out_point[1]} {out_point[2]} '
        f'-resampling 1 -resamplingstep 0.1 '
        #f'-ofile {temp_out}'
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


