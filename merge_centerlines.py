import vtk
from collections import defaultdict
import numpy as np
import pandas as pd
from individual_centerline import get_face_center
import os

def centerline_merging(branch_files, tol_high, tol_low, input_model_file, output_file, face_mapping, spatial_tolerance=0.2):

    # Phase 1: Union and cleaning in 2 phases
    aca_files = [f for f in branch_files if "ACA" in os.path.basename(f)]
    not_aca_files = [f for f in branch_files if "ACA" not in os.path.basename(f)]

    # General branch union (high tolerance)
    append_general = vtk.vtkAppendPolyData()
    for f in [f for f in not_aca_files if f not in aca_files]:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(f); reader.Update()
        append_general.AddInputData(reader.GetOutput())
    append_general.Update()

    cleaner_general = vtk.vtkCleanPolyData()
    cleaner_general.SetInputData(append_general.GetOutput())
    cleaner_general.SetTolerance(tol_high)
    cleaner_general.Update()

    # Final union with ACAs lower tolerance
    append_acas = vtk.vtkAppendPolyData()
    for f in aca_files:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(f); reader.Update()
        append_acas.AddInputData(reader.GetOutput())
    append_acas.Update()

    final_append = vtk.vtkAppendPolyData()
    final_append.AddInputData(cleaner_general.GetOutput())
    final_append.AddInputData(append_acas.GetOutput())
    final_append.Update()

    final_cleaner = vtk.vtkCleanPolyData()
    final_cleaner.SetInputData(final_append.GetOutput())
    final_cleaner.SetTolerance(tol_low) 
    final_cleaner.Update()

    # Clean duplicates
    clean_poly = final_cleaner.GetOutput()
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
    poly_unique.GetPointData().PassData(clean_poly.GetPointData()) 

    stripper = vtk.vtkStripper()
    stripper.SetInputData(poly_unique)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    poly_prepared = stripper.GetOutput()
    poly_prepared.BuildLinks()

    # Phase 2: Connectivity
    node_connectivity = defaultdict(int)
    for i in range(poly_prepared.GetNumberOfCells()):
        ids = poly_prepared.GetCell(i).GetPointIds()
        for j in range(ids.GetNumberOfIds() - 1):
            p1, p2 = ids.GetId(j), ids.GetId(j+1)
            node_connectivity[p1] += 1
            node_connectivity[p2] += 1
    bifurcations = {pid for pid, count in node_connectivity.items() if count >= 3}

    # Phase 3: Segmentation
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

    # Phase 4: Generate BranchIDs and Inflow/Outflow
    MASTER_INFLOWS = ["cap_L_ICA", "cap_R_ICA", "cap_L_VA", "cap_R_VA"]
    MASTER_OUTFLOWS = ["cap_L_SCA", "cap_L_PCA", "cap_L_MCA", "cap_L_ACA", "cap_R_ACA", "cap_R_MCA", "cap_R_PCA", "cap_R_SCA"]

    ID_OFFSET_OUTFLOWS = 4
    ID_OFFSET_INTERNAL = 12

    reader_mesh = vtk.vtkXMLPolyDataReader()
    reader_mesh.SetFileName(input_model_file)
    reader_mesh.Update()
    mesh_orig = reader_mesh.GetOutput()

    target_centers = {}
    print("\nFace center coordenates")
    for name in MASTER_INFLOWS + MASTER_OUTFLOWS:
        fid = face_mapping.get(name)
        if fid:
            c = get_face_center(mesh_orig, "ModelFaceID", fid)
            if c:
                target_centers[name] = np.array(c)
                print(f"FACE {fid:<3} | {name:<3} : ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")

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

        # Spatial diagnostic
        for name, center in target_centers.items():
            dist = min(np.linalg.norm(p_start - center), np.linalg.norm(p_end - center))
            if dist < min_d:
                min_d = dist
                target_match = name

        # Inflow and Outflow setting
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

    # Phase 5: Set Nodetype, UsageTag and Radius
    final_net.SetLines(final_cell_array)
    final_net.GetCellData().AddArray(branch_ids)

    # Transfer all the pointdata fields
    final_net.GetPointData().PassData(poly_prepared.GetPointData())

    # Create NodeType field
    node_type_array = vtk.vtkIntArray()
    node_type_array.SetName("NodeType")
    node_type_array.SetNumberOfTuples(final_net.GetNumberOfPoints())
    for i in range(final_net.GetNumberOfPoints()):
        node_type_array.SetTuple1(i, node_connectivity[i])
    final_net.GetPointData().AddArray(node_type_array)

    # Set UsageTag
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
                if np.linalg.norm(p - c) < 5: tag = 1; break
            if tag == 0:
                for c in out_centers:
                    if np.linalg.norm(p - c) < 5: tag = 2; break
        elif ntype > 2:
            tag = 3
        usage_tags.InsertNextValue(tag)
    final_net.GetPointData().AddArray(usage_tags)

    # Final save
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(final_net)
    writer.Write()

    print(f"\nEND")
    print(f"File saved in: {output_file}")
