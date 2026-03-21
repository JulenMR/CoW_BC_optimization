import vtk
import os
import glob
from collections import defaultdict

# --- CONFIGURACIÓN ---
path_base = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines"
output_file = os.path.join(path_base, "cow_full_final.vtp")
files = glob.glob(os.path.join(path_base, "fixed_tmp_cl_*"))

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
cleaner.SetTolerance(0.02) 
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

# GUARDAR
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(output_file)
writer.SetInputData(final_net)
writer.Write()

print(f"\n--- ÉXITO ---")
print(f"Líneas duplicadas purgadas: {count_removed}")
print(f"Ramas finales en el modelo: {final_net.GetNumberOfCells()}")