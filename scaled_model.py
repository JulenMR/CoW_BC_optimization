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
    
    new_lines = vtk.vtkCellArray()
    for i in range(num_pts - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        new_lines.InsertNextCell(line)
    
    new_poly = vtk.vtkPolyData()
    new_poly.SetPoints(new_points)
    new_poly.SetLines(new_lines)
    
    # Copiar radios si existen (Ojo: VMTK los suele llamar 'MaximumInscribedSphereRadius')
    old_radii = polydata.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if old_radii:
        new_poly.GetPointData().AddArray(old_radii)
        
    return new_poly

# --- NUEVA FUNCIÓN DE ESCALADO INTEGRADA ---
def scale_mesh_and_radii(polydata, factor=0.1):
    """Escala coordenadas y el array de radios de la malla."""
    transform = vtk.vtkTransform()
    transform.Scale(factor, factor, factor)
    
    t_filter = vtk.vtkTransformPolyDataFilter()
    t_filter.SetInputData(polydata)
    t_filter.SetTransform(transform)
    t_filter.Update()
    
    scaled_mesh = t_filter.GetOutput()
    
    # Escalar el array de radios si existe
    radii_name = "MaximumInscribedSphereRadius"
    radii = scaled_mesh.GetPointData().GetArray(radii_name)
    if radii:
        for i in range(radii.GetNumberOfTuples()):
            radii.SetTuple1(i, radii.GetTuple1(i) * factor)
    
    return scaled_mesh

# --- CONFIGURACIÓN ---
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models"
save_file = os.path.join(og_dir, "Centerlines/easy_example")
input_file = os.path.join(og_dir, "no_collaterals_reindexed.vtp")
cell_data_array = "ModelFaceID" 

inlet_ids = [18, 18] 
outlet_ids = [23, 19]

if not os.path.exists(save_file):
    os.makedirs(save_file)

# --- FASE 0: Carga y ESCALADO ---
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(input_file)
reader.Update()
raw_mesh = reader.GetOutput()

# ESCALAMOS AQUÍ (De mm a cm, por ejemplo)
print(f"--- Escalando malla original x0.1 ---")
mesh = scale_mesh_and_radii(raw_mesh, factor=0.1)

print("--- FASE 1: Extracción de segmentos individuales con integridad topográfica ---")
for i in range(len(inlet_ids)):
    in_id = inlet_ids[i]
    out_id = outlet_ids[i]
    
    # Estos centros ya saldrán en la escala correcta (x0.1)
    in_point = get_face_center(mesh, cell_data_array, in_id)
    out_point = get_face_center(mesh, cell_data_array, out_id)
    
    if in_point is None or out_point is None:
        print(f"Error: No se encontraron las caras {in_id} o {out_id}")
        continue

    fixed_temp_out = os.path.join(save_file, f"fixed_tmp_cl_{in_id}_{out_id}.vtp")
    
    # Al pasar los puntos ya escalados a VMTK, la centerline se extraerá en la escala correcta
    # IMPORTANTE: Ajustamos el 'resamplingstep' a 0.01 si hemos escalado x0.1 para mantener resolución
    vmtk_cmd = (
        f'vmtkcenterlines -ifile {input_file} ' # Ojo: VMTK leerá el archivo, pero usará los puntos escalados
        f'-seedselector pointlist '
        f'-sourcepoints {in_point[0]} {in_point[1]} {in_point[2]} '
        f'-targetpoints {out_point[0]} {out_point[1]} {out_point[2]} '
        f'-resampling 1 -resamplingstep 0.01 ' 
    )
    
    print(f"\n>> Procesando camino: {in_id} a {out_id} (Escalado)")
    myPype = pypes.PypeRun(vmtk_cmd)

    cl_obj = myPype.GetScriptObject('vmtkcenterlines', '0').Centerlines

    if cl_obj:
        # Dado que vmtkcenterlines leyó el 'input_file' original (grande), 
        # pero usó semillas pequeñas, a veces el output puede ser confuso.
        # Por seguridad, escalamos el objeto resultante de la centerline también:
        cl_obj_scaled = scale_mesh_and_radii(cl_obj, factor=0.1)
        
        # RECONSTRUCCIÓN
        fixed_cl = rebuild_polydata_from_scratch(cl_obj_scaled)
        
        # GUARDADO
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(fixed_temp_out)
        writer.SetInputData(fixed_cl)
        writer.Write()
        print(f"[OK] Centerline escalada y guardada en {fixed_temp_out}")