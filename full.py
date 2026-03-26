import vtk
import os
import glob
from collections import defaultdict
import numpy as np


def get_face_center(polydata, array_name, face_id):
    """Filtra la malla y devuelve las coordenadas (x,y,z) del centro de la cara."""
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(polydata)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, array_name)
    threshold.SetLowerThreshold(face_id)
    threshold.SetUpperThreshold(face_id)
    threshold.Update()
    if threshold.GetOutput().GetNumberOfCells() == 0: return None
    center_filter = vtk.vtkCenterOfMass()
    center_filter.SetInputData(threshold.GetOutput())
    center_filter.SetUseScalarsAsWeights(False)
    center_filter.Update()
    return center_filter.GetCenter()

# --- CONFIGURACIÓN ---
path_base = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines/medium_example"
output_file = os.path.join(path_base, "cow_medium_final.vtp")
files = glob.glob(os.path.join(path_base, "fixed_tmp_cl_*"))
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models"
save_file = os.path.join(og_dir, "Centerlines/easy_example")
mesh_original_path = os.path.join(og_dir, "no_collaterals_reindexed.vtp")
cell_data_array = "ModelFaceID" 
#inflow_ids = [12, 14, 16, 18]
#outflow_ids = [13, 15, 17, 19, 20, 21, 22, 23]
inflow_ids = [12, 16]
outflow_ids = [15, 19, 21, 20]

if not files:
    print("No se encontraron archivos."); exit()

# 1. Phase of merging (Append + Clean + Stripper)
# ---------------------------------------------------------
append_filter = vtk.vtkAppendPolyData()
for f in files:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(f); reader.Update()
    append_filter.AddInputData(reader.GetOutput())
append_filter.Update()

# Cleaner Fuses points that are within a range 
cleaner = vtk.vtkCleanPolyData()
cleaner.SetInputData(append_filter.GetOutput())
cleaner.SetTolerance(0.025) 
cleaner.PointMergingOn()
cleaner.Update()

# Stripper: Combines simple lines defined by 2 points into a polyline
stripper = vtk.vtkStripper()
stripper.SetInputData(cleaner.GetOutput())
stripper.JoinContiguousSegmentsOn()
stripper.Update()

poly_prepared = stripper.GetOutput()
poly_prepared.BuildLinks()

# 2. Topological phase (Connectivity map and bifurcations)
# ---------------------------------------------------------
unique_segments = set()
node_connectivity = defaultdict(int)

# Bifurcations are identified as nodes with various polylines 
for i in range(poly_prepared.GetNumberOfCells()):
    ids = poly_prepared.GetCell(i).GetPointIds()
    for j in range(ids.GetNumberOfIds() - 1):
        p1, p2 = ids.GetId(j), ids.GetId(j+1)
        segment = tuple(sorted((p1, p2)))
        if segment not in unique_segments:
            unique_segments.add(segment)
            node_connectivity[p1] += 1
            node_connectivity[p2] += 1

bifurcations = {pid for pid, count in node_connectivity.items() if count >= 3}
print(f"Number of bifurcations: {len(bifurcations)}")

# 3. Segmentation phase (Cutting polyline at Junctions)
# ---------------------------------------------------------
segmented_cells = [] # Lista temporal para guardar las listas de IDs

for i in range(poly_prepared.GetNumberOfCells()):
    ids = poly_prepared.GetCell(i).GetPointIds()
    current_branch = []
    for j in range(ids.GetNumberOfIds()):
        pid = ids.GetId(j)
        current_branch.append(pid)
        # Cortar si es una bifurcación y no es el primer punto
        if pid in bifurcations and len(current_branch) > 1:
            segmented_cells.append(list(current_branch))
            current_branch = [pid]
    # Añadir el resto de la línea
    if len(current_branch) > 1:
        segmented_cells.append(list(current_branch))

# 4. Cleaning phase. Delete duplicated polylines
# ---------------------------------------------------------
final_cell_array = vtk.vtkCellArray()
seen_signatures = set()
count_removed = 0

for branch_pts in segmented_cells:
    # Firma: (inicio, fin) ordenados. Si necesitas más rigor, usa tuple(sorted(branch_pts))
    signature = tuple(sorted((branch_pts[0], branch_pts[-1])))
    
    if signature not in seen_signatures:
        seen_signatures.add(signature)
        poly_line = vtk.vtkPolyLine()
        poly_line.GetPointIds().SetNumberOfIds(len(branch_pts))
        for idx, p_id in enumerate(branch_pts):
            poly_line.GetPointIds().SetId(idx, p_id)
        final_cell_array.InsertNextCell(poly_line)
    else:
        count_removed += 1

# 5. FInal ensambling
# ---------------------------------------------------------
final_net = vtk.vtkPolyData()
final_net.SetPoints(poly_prepared.GetPoints())
final_net.SetLines(final_cell_array)

# BranchID correlativo
branch_ids = vtk.vtkIntArray()
branch_ids.SetName("BranchID")
for i in range(final_net.GetNumberOfCells()):
    branch_ids.InsertNextValue(i)
final_net.GetCellData().AddArray(branch_ids)

# NodeType (basado en conectividad real)
node_type = vtk.vtkIntArray()
node_type.SetName("NodeType")
node_type.SetNumberOfTuples(final_net.GetNumberOfPoints())
for i in range(final_net.GetNumberOfPoints()):
    node_type.SetTuple1(i, node_connectivity[i])
final_net.GetPointData().AddArray(node_type)

# Traspaso de PointData (radios, etc.)
final_net.GetPointData().PassData(poly_prepared.GetPointData())

# 6. FASE DE TAGEO ESPACIAL (Inflow / Outflow)
# ---------------------------------------------------------
print("--- FASE 6: Etiquetado espacial de flujos ---")

# Cargar malla original para obtener centros exactos
reader_mesh = vtk.vtkXMLPolyDataReader()
reader_mesh.SetFileName(mesh_original_path)
reader_mesh.Update()
mesh_orig = reader_mesh.GetOutput()

# Obtener coordenadas reales
inflow_coords = [get_face_center(mesh_orig, "ModelFaceID", fid) for fid in inflow_ids]
outflow_coords = [get_face_center(mesh_orig, "ModelFaceID", fid) for fid in outflow_ids]

# Crear Array de Tags
usage_tags = vtk.vtkIntArray()
usage_tags.SetName("UsageTag") # 1: Inflow, 2: Outflow, 0: Internal

points = final_net.GetPoints()
# Usamos 0.2 ya que tu distancia mínima detectada fue 0.1452
tolerance = 0.33
node_types = final_net.GetPointData().GetArray("NodeType")

tolerance = 0.5 
count_inflow = 0
count_outflow = 0
count_joints = 0

for i in range(final_net.GetNumberOfPoints()):
    p = np.array(points.GetPoint(i))
    
    # CORRECTO: Extraemos el valor entero del punto 'i'
    ntype = int(node_types.GetTuple1(i)) 
    
    tag = 0
    
    # CASO A: Es un extremo (NodeType 1) -> Buscamos Inflow o Outflow
    if ntype == 1:
        # ¿Es este punto un INFLOW real?
        for c in inflow_coords:
            if c and np.linalg.norm(p - np.array(c)) < tolerance:
                tag = 1
                count_inflow += 1
                break
        
        # ¿Es este punto un OUTFLOW real? (solo si no se marcó como inflow)
        if tag == 0:
            for c in outflow_coords:
                if c and np.linalg.norm(p - np.array(c)) < tolerance:
                    tag = 2
                    count_outflow += 1
                    break
    
    # CASO B: Es una bifurcación o unión (Joint)
    # ntype > 2 significa que 3 o más líneas se encuentran ahí
    elif ntype > 2:
        tag = 3
        count_joints += 1
    
    usage_tags.InsertNextValue(tag)

final_net.GetPointData().AddArray(usage_tags)

print(f"\n--- ETIQUETADO COMPLETADO ---")
print(f"Inflows (Tag 1): {count_inflow}")
print(f"Outflows (Tag 2): {count_outflow}")
print(f"Joints   (Tag 3): {count_joints}")

# 7. GUARDADO FINAL
# ---------------------------------------------------------
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(output_file)
writer.SetInputData(final_net)
writer.Write()

print(f"\n--- ÉXITO ---")
print(f"Archivo final guardado en: {output_file}")
# Usamos las variables que incrementamos en el bucle para evitar el error de iteración
print(f"Puntos etiquetados como Inflow: {count_inflow}")
print(f"Puntos etiquetados como Outflow: {count_outflow}")