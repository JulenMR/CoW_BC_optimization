import vtk
import os
import glob
from collections import defaultdict
import numpy as np
import pandas as pd

def get_face_center(polydata, array_name, face_id):
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
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models"
centerline_file = os.path.join(og_dir, "FULL_PIPELINE")
output_file = os.path.join(centerline_file, "CoW_centerline_final_clean.vtp")
files = glob.glob(os.path.join(centerline_file, "fixed_tmp_cl_*"))
mesh_original_path = os.path.join(centerline_file, "scaled.vtp")
xml_file = os.path.join(og_dir, "no_collaterals.mdl")

tolerance_cleaning = 0.01
spatial_tolerance = 0.2 

# --- MAPEO DE CARAS ---
df_faceID = pd.read_xml(xml_file, xpath=".//face", parser="etree")
df_caps = df_faceID[df_faceID['type'] == 'cap']
face_mapping = dict(zip(df_caps['name'], df_caps['id'].astype(int)))

# --- FASE 1: UNIÓN Y LIMPIEZA AGRESIVA (ANTI-DUPLICADOS) ---
append_filter = vtk.vtkAppendPolyData()
for f in files:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(f); reader.Update()
    append_filter.AddInputData(reader.GetOutput())
append_filter.Update()

cleaner = vtk.vtkCleanPolyData()
cleaner.SetInputData(append_filter.GetOutput())
cleaner.SetTolerance(tolerance_cleaning) 
cleaner.PointMergingOn()
cleaner.Update()

clean_poly = cleaner.GetOutput()
unique_cells = vtk.vtkCellArray()
existing_segments = set()

for i in range(clean_poly.GetNumberOfCells()):
    ids = clean_poly.GetCell(i).GetPointIds()
    for j in range(ids.GetNumberOfIds() - 1):
        p1, p2 = ids.GetId(j), ids.GetId(j+1)
        segment = tuple(sorted((p1, p2)))
        if segment not in existing_segments:
            existing_segments.add(segment)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, p1)
            line.GetPointIds().SetId(1, p2)
            unique_cells.InsertNextCell(line)

poly_unique = vtk.vtkPolyData()
poly_unique.SetPoints(clean_poly.GetPoints())
poly_unique.SetLines(unique_cells)
poly_unique.GetPointData().PassData(clean_poly.GetPointData()) # Mantiene Radios originales

stripper = vtk.vtkStripper()
stripper.SetInputData(poly_unique)
stripper.JoinContiguousSegmentsOn()
stripper.Update()

poly_prepared = stripper.GetOutput()
poly_prepared.BuildLinks()

# --- FASE 2: TOPOLOGÍA ---
node_connectivity = defaultdict(int)
for i in range(poly_prepared.GetNumberOfCells()):
    ids = poly_prepared.GetCell(i).GetPointIds()
    for j in range(ids.GetNumberOfIds() - 1):
        p1, p2 = ids.GetId(j), ids.GetId(j+1)
        node_connectivity[p1] += 1
        node_connectivity[p2] += 1
bifurcations = {pid for pid, count in node_connectivity.items() if count >= 3}

# --- FASE 3: SEGMENTACIÓN ---
segmented_cells = [] 
for i in range(poly_prepared.GetNumberOfCells()):
    ids = poly_prepared.GetCell(i).GetPointIds()
    current_branch = []
    for j in range(ids.GetNumberOfIds()):
        pid = ids.GetId(j)
        current_branch.append(pid)
        if pid in bifurcations and len(current_branch) > 1:
            segmented_cells.append(list(current_branch))
            current_branch = [pid]
    if len(current_branch) > 1:
        segmented_cells.append(list(current_branch))

# --- FASE 4 Y 5: CLASIFICACIÓN CON DEBUG ---
MASTER_INFLOWS = ["cap_Left_ICA_MCA", "cap_Right_ICA_MCA", "cap_LeftVert_Basilar_LeftPost", "cap_RightVert"]
MASTER_OUTFLOWS = ["cap_Left_SCA", "cap_LeftVert_Basilar_LeftPost_2", "cap_Left_ICA_MCA_2", "cap_Left_Anterior", "cap_Right_Anterior", "cap_Right_ICA_MCA_2", "cap_Right_Post", "cap_Right_SCA"]

ID_OFFSET_OUTFLOWS = 4
ID_OFFSET_INTERNAL = 12

reader_mesh = vtk.vtkXMLPolyDataReader()
reader_mesh.SetFileName(mesh_original_path)
reader_mesh.Update()
mesh_orig = reader_mesh.GetOutput()

target_centers = {}
print("\n--- COORDENADAS CENTROS CARAS ---")
for name in MASTER_INFLOWS + MASTER_OUTFLOWS:
    fid = face_mapping.get(name)
    if fid:
        c = get_face_center(mesh_orig, "ModelFaceID", fid)
        if c:
            target_centers[name] = np.array(c)
            print(f"FACE {fid:<3} | {name:<30} : ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")

final_net = vtk.vtkPolyData()
final_net.SetPoints(poly_prepared.GetPoints())
final_cell_array = vtk.vtkCellArray()
branch_ids = vtk.vtkIntArray()
branch_ids.SetName("BranchID")

internal_count = 0
seen_signatures = set()

for b_idx, branch_pts in enumerate(segmented_cells):
    signature = tuple(sorted((branch_pts[0], branch_pts[-1])))
    if signature in seen_signatures: continue
    seen_signatures.add(signature)

    p_start = np.array(poly_prepared.GetPoints().GetPoint(branch_pts[0]))
    p_end = np.array(poly_prepared.GetPoints().GetPoint(branch_pts[-1]))
    
    assigned_id = None
    min_d = float('inf')
    target_match = ""

    # Diagnóstico espacial
    for name, center in target_centers.items():
        dist = min(np.linalg.norm(p_start - center), np.linalg.norm(p_end - center))
        if dist < min_d:
            min_d = dist
            target_match = name

    # Asignación
    for idx, name in enumerate(MASTER_INFLOWS):
        if name in target_centers:
            if min(np.linalg.norm(p_start - target_centers[name]), np.linalg.norm(p_end - target_centers[name])) < spatial_tolerance:
                assigned_id = idx; break
    if assigned_id is None:
        for idx, name in enumerate(MASTER_OUTFLOWS):
            if name in target_centers:
                if min(np.linalg.norm(p_start - target_centers[name]), np.linalg.norm(p_end - target_centers[name])) < spatial_tolerance:
                    assigned_id = idx + ID_OFFSET_OUTFLOWS; break

    if assigned_id is None:
        assigned_id = internal_count + ID_OFFSET_INTERNAL
        internal_count += 1

    poly_line = vtk.vtkPolyLine()
    poly_line.GetPointIds().SetNumberOfIds(len(branch_pts))
    for i, p_id in enumerate(branch_pts): poly_line.GetPointIds().SetId(i, p_id)
    final_cell_array.InsertNextCell(poly_line)
    branch_ids.InsertNextValue(assigned_id)

# --- FASE 6: NODETYPE, USAGETAG Y RADIUS ---
final_net.SetLines(final_cell_array)
final_net.GetCellData().AddArray(branch_ids)

# 1. Traspasar todos los arrays de puntos (incluye MaximumInscribedSphereRadius)
final_net.GetPointData().PassData(poly_prepared.GetPointData())

# 2. NodeType
node_type_array = vtk.vtkIntArray()
node_type_array.SetName("NodeType")
node_type_array.SetNumberOfTuples(final_net.GetNumberOfPoints())
for i in range(final_net.GetNumberOfPoints()):
    node_type_array.SetTuple1(i, node_connectivity[i])
final_net.GetPointData().AddArray(node_type_array)

# 3. UsageTag
usage_tags = vtk.vtkIntArray()
usage_tags.SetName("UsageTag")
in_centers = [target_centers[n] for n in MASTER_INFLOWS if n in target_centers]
out_centers = [target_centers[n] for n in MASTER_OUTFLOWS if n in target_centers]

for i in range(final_net.GetNumberOfPoints()):
    p = np.array(final_net.GetPoints().GetPoint(i))
    ntype = node_type_array.GetTuple1(i)
    tag = 0
    if ntype == 1:
        for c in in_centers:
            if np.linalg.norm(p - c) < 0.5: tag = 1; break
        if tag == 0:
            for c in out_centers:
                if np.linalg.norm(p - c) < 0.5: tag = 2; break
    elif ntype > 2:
        tag = 3
    usage_tags.InsertNextValue(tag)
final_net.GetPointData().AddArray(usage_tags)

# --- GUARDADO FINAL ---
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(output_file)
writer.SetInputData(final_net)
writer.Write()

print(f"\n--- ÉXITO ---")
print(f"Archivo guardado en: {output_file}")
print(f"Ramas internas creadas: {internal_count}")