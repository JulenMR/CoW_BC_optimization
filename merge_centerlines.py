import os
import vtk
import numpy as np
from collections import defaultdict
from individual_centerline import get_face_center

def centerline_merging(branch_files, tol_general, tol_aca, input_model_file, output_file, face_mapping, spatial_tolerance=1.0, min_branch_length=0.5):
    
    ### Phase 0: Load and Merge Centerlines 
    aca_files = [f for f in branch_files if "ACA" in os.path.basename(f)]
    not_aca_files = [f for f in branch_files if "ACA" not in os.path.basename(f)]

    # Clean general branches
    append_general = vtk.vtkAppendPolyData()
    for f in not_aca_files:
        if os.path.exists(f):
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(f)
            reader.Update()
            append_general.AddInputData(reader.GetOutput())
    append_general.Update()

    cleaner_general = vtk.vtkCleanPolyData()
    cleaner_general.SetInputData(append_general.GetOutput())
    cleaner_general.SetToleranceIsAbsolute(True)
    cleaner_general.SetAbsoluteTolerance(tol_general)
    cleaner_general.PointMergingOn()
    cleaner_general.Update()

    # Clean ACA branches with finer absolute tolerance
    append_acas = vtk.vtkAppendPolyData()
    for f in aca_files:
        if os.path.exists(f):
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(f)
            reader.Update()
            append_acas.AddInputData(reader.GetOutput())
    append_acas.Update()

    final_append = vtk.vtkAppendPolyData()
    final_append.AddInputData(cleaner_general.GetOutput())
    final_append.AddInputData(append_acas.GetOutput())
    final_append.Update()

    final_cleaner = vtk.vtkCleanPolyData()
    final_cleaner.SetInputData(final_append.GetOutput())
    final_cleaner.SetToleranceIsAbsolute(True)
    final_cleaner.SetAbsoluteTolerance(tol_aca)
    final_cleaner.PointMergingOn()
    final_cleaner.Update()

    clean_poly = final_cleaner.GetOutput()
    num_points = clean_poly.GetNumberOfPoints()
    points = clean_poly.GetPoints()

    ### Phase 1: Construct Graph without Duplicate Edges
    unique_edges = set()
    graph = defaultdict(set)

    for i in range(clean_poly.GetNumberOfCells()):
        cell = clean_poly.GetCell(i)
        ids = cell.GetPointIds()
        for j in range(ids.GetNumberOfIds() - 1):
            p1 = ids.GetId(j)
            p2 = ids.GetId(j + 1)
            if p1 == p2:
                continue
            edge = tuple(sorted((p1, p2)))
            if edge not in unique_edges:
                unique_edges.add(edge)
                graph[p1].add(p2)
                graph[p2].add(p1)

    ### Phase 2: Topological Node Identification
    node_degree = {p: len(neighbors) for p, neighbors in graph.items()}
    junctions_and_terminals = {p for p, deg in node_degree.items() if deg != 2}

    ### Phase 3: Extract Branches and Filter Microsegments
    visited_edges = set()
    raw_branches = []

    for start_node in junctions_and_terminals:
        for neighbor in graph[start_node]:
            edge = tuple(sorted((start_node, neighbor)))
            if edge in visited_edges:
                continue

            branch = [start_node, neighbor]
            visited_edges.add(edge)
            curr, prev = neighbor, start_node

            while node_degree[curr] == 2:
                next_nodes = graph[curr] - {prev}
                if not next_nodes:
                    break
                next_node = next_nodes.pop()
                next_edge = tuple(sorted((curr, next_node)))

                branch.append(next_node)
                visited_edges.add(next_edge)
                prev, curr = curr, next_node

            raw_branches.append(branch)

    # Filter out microsegments shorter than min_branch_length (in mm)
    branches = []
    for branch in raw_branches:
        length = sum(
            np.linalg.norm(
                np.array(points.GetPoint(branch[k])) - 
                np.array(points.GetPoint(branch[k + 1]))
            )
            for k in range(len(branch) - 1)
        )
        if length >= min_branch_length or len(branch) > 3:
            branches.append(branch)

    ### Phase 4: Map Boundary Faces, Assign BranchIDs, and Endpoints
    MASTER_INFLOWS = ["cap_L_ICA", "cap_R_ICA", "cap_L_VA", "cap_R_VA"]
    MASTER_OUTFLOWS = [
        "cap_L_SCA", "cap_L_PCA", "cap_L_MCA", "cap_L_ACA", 
        "cap_R_ACA", "cap_R_MCA", "cap_R_PCA", "cap_R_SCA"     
    ]

    reader_mesh = vtk.vtkXMLPolyDataReader()
    reader_mesh.SetFileName(input_model_file)
    reader_mesh.Update()
    mesh_orig = reader_mesh.GetOutput()

    target_centers = {}
    for name in MASTER_INFLOWS + MASTER_OUTFLOWS:
        fid = face_mapping.get(name)
        if fid:
            c = get_face_center(mesh_orig, "ModelFaceID", fid)
            if c:
                target_centers[name] = np.array(c)

    print(f"Found and mapped {len(target_centers)} boundary face centers ")

    branch_ids_point = vtk.vtkIntArray()
    branch_ids_point.SetName("BranchID")
    branch_ids_point.SetNumberOfTuples(num_points)
    for i in range(num_points):
        branch_ids_point.SetTuple1(i, -1)

    branch_ids_cell = vtk.vtkIntArray()
    branch_ids_cell.SetName("BranchID")

    final_cell_array = vtk.vtkCellArray()
    internal_count = 0

    for branch_pts in branches:
        p_start_idx = branch_pts[0]
        p_end_idx = branch_pts[-1]

        p_start = np.array(points.GetPoint(p_start_idx))
        p_end = np.array(points.GetPoint(p_end_idx))
        assigned_id = None

        # Assign Inflows (IDs 0-3) and snap endpoint to exact face center
        for idx, name in enumerate(MASTER_INFLOWS):
            if name in target_centers:
                d_start = np.linalg.norm(p_start - target_centers[name])
                d_end = np.linalg.norm(p_end - target_centers[name])
                if d_start < spatial_tolerance or d_end < spatial_tolerance:
                    assigned_id = idx
                    if d_start < d_end:
                        points.SetPoint(p_start_idx, target_centers[name])
                    else:
                        points.SetPoint(p_end_idx, target_centers[name])
                    break

        # Assign Outflows (IDs 4-12) and snap endpoint to exact face center
        if assigned_id is None:
            for idx, name in enumerate(MASTER_OUTFLOWS):
                if name in target_centers:
                    d_start = np.linalg.norm(p_start - target_centers[name])
                    d_end = np.linalg.norm(p_end - target_centers[name])
                    if d_start < spatial_tolerance or d_end < spatial_tolerance:
                        assigned_id = idx + 4
                        if d_start < d_end:
                            points.SetPoint(p_start_idx, target_centers[name])
                        else:
                            points.SetPoint(p_end_idx, target_centers[name])
                        break

        # Assign Internal branches 
        if assigned_id is None:
            assigned_id = internal_count + 12
            internal_count += 1

        poly_line = vtk.vtkPolyLine()
        poly_line.GetPointIds().SetNumberOfIds(len(branch_pts))
        for j, p_id in enumerate(branch_pts):
            poly_line.GetPointIds().SetId(j, p_id)
            branch_ids_point.SetTuple1(p_id, assigned_id)

        final_cell_array.InsertNextCell(poly_line)
        branch_ids_cell.InsertNextValue(assigned_id)

    ### Phase 5: Build Final PolyData, Attach Attributes, and Export
    final_net = vtk.vtkPolyData()
    final_net.SetPoints(points)
    final_net.SetLines(final_cell_array)
    final_net.GetCellData().AddArray(branch_ids_cell)
    final_net.GetPointData().PassData(clean_poly.GetPointData())
    final_net.GetPointData().AddArray(branch_ids_point)

    node_type_array = vtk.vtkIntArray()
    node_type_array.SetName("NodeType")
    node_type_array.SetNumberOfTuples(num_points)
    for i in range(num_points):
        node_type_array.SetTuple1(i, node_degree.get(i, 0))
    final_net.GetPointData().AddArray(node_type_array)

    usage_tags = vtk.vtkIntArray()
    usage_tags.SetName("UsageTag")
    in_centers = [target_centers[n] for n in MASTER_INFLOWS if n in target_centers]
    out_centers = [target_centers[n] for n in MASTER_OUTFLOWS if n in target_centers]

    for i in range(num_points):
        p = np.array(final_net.GetPoints().GetPoint(i))
        deg = node_degree.get(i, 0)
        tag = 0
        if deg == 1:
            if any(np.linalg.norm(p - c) < spatial_tolerance for c in in_centers):
                tag = 1
            elif any(np.linalg.norm(p - c) < spatial_tolerance for c in out_centers):
                tag = 2
        elif deg >= 3:
            tag = 3
        usage_tags.InsertNextValue(tag)

    final_net.GetPointData().AddArray(usage_tags)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(final_net)
    writer.Write()

    print(f"Merged centerline network successfully saved to: {output_file}")