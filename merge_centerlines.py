import vtk
from collections import defaultdict
import numpy as np
import pandas as pd
from individual_centerline import get_face_center
import os

def centerline_merging(branch_files, tol_general, tol_aca, input_model_file, output_file, face_mapping, spatial_tolerance=0.2):
    ### Phase 0: Get outlet points from each centerline
    ending_nodes = []
    for f in branch_files:
        if os.path.exists(f):
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(f); reader.Update()
            poly_branch = reader.GetOutput()
            
            # Count how many cells are in contact
            local_connectivity = defaultdict(int)
            for i in range(poly_branch.GetNumberOfCells()):
                cell_ids = poly_branch.GetCell(i).GetPointIds()
                for j in range(cell_ids.GetNumberOfIds()):
                    local_connectivity[cell_ids.GetId(j)] += 1
            
            # Outlet are points that only touch 1 cell
            for pid, count in local_connectivity.items():
                if count == 1:
                    ending_nodes.append(poly_branch.GetPoint(pid))
                    
    # Eliminate duplicates
    ending_nodes = list(set(tuple(np.round(p, 6)) for p in ending_nodes))
    print(f"{len(ending_nodes)} outlets detected")

    ### Phase 1: Union and cleaning in 2 iterations
    aca_files = [f for f in branch_files if "ACA" in os.path.basename(f)]
    not_aca_files = [f for f in branch_files if "ACA" not in os.path.basename(f)]

        # Unify general branches
    append_general = vtk.vtkAppendPolyData()
    for f in [f for f in not_aca_files if f not in aca_files]:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(f); reader.Update()
        append_general.AddInputData(reader.GetOutput())
    append_general.Update()
        # Clean with general tolerance
    cleaner_general = vtk.vtkCleanPolyData()
    cleaner_general.SetInputData(append_general.GetOutput())
    cleaner_general.SetTolerance(tol_general)
    cleaner_general.Update()

        # Now append ACA branches
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
        # Use a finer tolerance for ACAs
    final_cleaner = vtk.vtkCleanPolyData()
    final_cleaner.SetInputData(final_append.GetOutput())
    final_cleaner.SetTolerance(tol_aca) 
    final_cleaner.Update()

    clean_poly = final_cleaner.GetOutput()

    existing_points = clean_poly.GetPoints()
    point_data = clean_poly.GetPointData()
    
    num_original_points = clean_poly.GetNumberOfPoints()
    
    # Manually adds ending nodes to the clean centerline
    outlet_to_id = {}
    points_to_connect = [] 
    
    for pt_orig in ending_nodes:
        found = False
        for i in range(num_original_points):
            # Compares distance of each end node with each centerline point
            pt_act = clean_poly.GetPoint(i)
            dist = ((pt_act[0] - pt_orig[0])**2 + (pt_act[1] - pt_orig[1])**2 + (pt_act[2] - pt_orig[2])**2)**0.5
            
            if dist < spatial_tolerance:
                new_id = existing_points.InsertNextPoint(pt_orig)
                outlet_to_id[pt_orig] = new_id
                # If a point is inside threshold distance, it saves the tuple (ending node+close point)
                points_to_connect.append((i, new_id))
                # Saves all the array data
                for k in range(point_data.GetNumberOfArrays()):
                    arr = point_data.GetArray(k)
                    if arr is not None:
                        arr.InsertNextTuple(arr.GetTuple(i))
                        
                found = True
                break
        
        if not found:
            new_id = existing_points.InsertNextPoint(pt_orig)
            outlet_to_id[pt_orig] = new_id
            for k in range(point_data.GetNumberOfArrays()):
                arr = point_data.GetArray(k)
                if arr is not None:
                    arr.InsertNextTuple(arr.GetTuple(0))

    # Deletes duplicated segments
    unique_cells = vtk.vtkCellArray()
    existing_segments = set()
    for i in range(clean_poly.GetNumberOfCells()):
        ids = clean_poly.GetCell(i).GetPointIds()
        n_ids = ids.GetNumberOfIds()
        
        for j in range(n_ids - 1):
            p1, p2 = ids.GetId(j), ids.GetId(j+1)
            if p1 == p2: continue  

            segment = tuple(sorted((p1, p2)))
            if segment not in existing_segments:
                existing_segments.add(segment)
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, p1)
                line.GetPointIds().SetId(1, p2)
                unique_cells.InsertNextCell(line)

    # Creates new links to add the ending node to its closest point
    for p_intern, p_cap in points_to_connect:
        segment = tuple(sorted((p_intern, p_cap)))
        if segment not in existing_segments:
            existing_segments.add(segment)
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, p_intern)
            line.GetPointIds().SetId(1, p_cap)
            unique_cells.InsertNextCell(line)

    # Reconstructs cell connections
    poly_unique = vtk.vtkPolyData()
    poly_unique.SetPoints(existing_points)
    poly_unique.SetLines(unique_cells)
    poly_unique.GetPointData().PassData(clean_poly.GetPointData()) 

    stripper = vtk.vtkStripper()
    stripper.SetInputData(poly_unique)
    stripper.JoinContiguousSegmentsOn()
    stripper.Update()

    poly_prepared = stripper.GetOutput()
    poly_prepared.BuildLinks()

    ### Phase 2: Connectivity
    node_connectivity = defaultdict(int)
    for i in range(poly_prepared.GetNumberOfCells()):
        ids = poly_prepared.GetCell(i).GetPointIds()
        for j in range(ids.GetNumberOfIds() - 1):
            p1, p2 = ids.GetId(j), ids.GetId(j+1)
            node_connectivity[p1] += 1
            node_connectivity[p2] += 1
    # If a node has > 3 connections, it is a bifurcation
    bifurcations = {pid for pid, count in node_connectivity.items() if count >= 3}

    ### Phase 3: Segmentation
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

    ### Phase 4: Generate BranchIDs and Inflow/Outflow
    MASTER_INFLOWS = ["cap_L_ICA", "cap_R_ICA", "cap_L_VA", "cap_R_VA"]
    MASTER_OUTFLOWS = ["cap_L_SCA", "cap_L_PCA", "cap_L_MCA", "cap_L_ACA", "cap_R_ACA", "cap_R_MCA", "cap_R_PCA", "cap_R_SCA"]

    ID_OFFSET_OUTFLOWS = 4
    ID_OFFSET_INTERNAL = 12

    reader_mesh = vtk.vtkXMLPolyDataReader()
    reader_mesh.SetFileName(input_model_file)
    reader_mesh.Update()
    mesh_orig = reader_mesh.GetOutput()

    # Calculates the center for each cap in the original mesh
    target_centers = {}
    print("\nFace center coordinates")
    for name in MASTER_INFLOWS + MASTER_OUTFLOWS:
        fid = face_mapping.get(name)
        if fid:
            c = get_face_center(mesh_orig, "ModelFaceID", fid)
            if c:
                target_centers[name] = np.array(c)
                print(f"FACE {fid:<3} | {name:<3} : ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")


    branch_ids_point = vtk.vtkIntArray()
    branch_ids_point.SetName("BranchID")
    branch_ids_point.SetNumberOfTuples(poly_prepared.GetNumberOfPoints())
    for i in range(poly_prepared.GetNumberOfPoints()):
        branch_ids_point.SetTuple1(i, -1)

    internal_count = 0
    segments_by_id = defaultdict(list)
    # Iterates through all branches, checks if the starting or ending node of the branch falls in the threshold of the caps center points
    for branch_pts in segmented_cells:
        p_start = np.array(poly_prepared.GetPoints().GetPoint(branch_pts[0]))
        p_end = np.array(poly_prepared.GetPoints().GetPoint(branch_pts[-1]))
        assigned_id = None

        # Assign inflows
        for idx, name in enumerate(MASTER_INFLOWS):
            if name in target_centers:
                dist = min(np.linalg.norm(p_start - target_centers[name]), np.linalg.norm(p_end - target_centers[name]))
                if dist < spatial_tolerance:
                    assigned_id = idx; break
        
        # Assign outflows
        if assigned_id is None:
            for idx, name in enumerate(MASTER_OUTFLOWS):
                if name in target_centers:
                    dist = min(np.linalg.norm(p_start - target_centers[name]), np.linalg.norm(p_end - target_centers[name]))
                    if dist < spatial_tolerance:
                        assigned_id = idx + ID_OFFSET_OUTFLOWS; break

        # Assign intermediate point
        if assigned_id is None:
            assigned_id = internal_count + ID_OFFSET_INTERNAL
            internal_count += 1

        # Save all points by ID
        segments_by_id[assigned_id].append(branch_pts)

    # Merge all segments by ID
    final_cell_array = vtk.vtkCellArray()
    branch_ids_cell = vtk.vtkIntArray()
    branch_ids_cell.SetName("BranchID")

    for b_id, sub_segments in segments_by_id.items():
        if not sub_segments:
            continue
            
        merged_path = list(sub_segments[0])
        unused_segments = sub_segments[1:]
        
        # Finds the correct position of the segments and inverts them if necessary
        while unused_segments:
            progress = False
            for i, seg in enumerate(unused_segments):
                if merged_path[-1] == seg[0]:
                    merged_path.extend(seg[1:])
                    unused_segments.pop(i); progress = True; break
                elif merged_path[0] == seg[-1]:
                    merged_path = seg[:-1] + merged_path
                    unused_segments.pop(i); progress = True; break
                elif merged_path[-1] == seg[-1]: 
                    merged_path.extend(reversed(seg[:-1]))
                    unused_segments.pop(i); progress = True; break
                elif merged_path[0] == seg[0]: 
                    merged_path = list(reversed(seg[1:])) + merged_path
                    unused_segments.pop(i); progress = True; break
            
            if not progress:
                # This means that segments with the same ID are physically disconnected
                merged_path.extend(unused_segments[0])
                unused_segments.pop(0)

        # Create unique polyline
        poly_line = vtk.vtkPolyLine()
        poly_line.GetPointIds().SetNumberOfIds(len(merged_path))
        
        for i, p_id in enumerate(merged_path): 
            poly_line.GetPointIds().SetId(i, p_id)
            branch_ids_point.SetTuple1(p_id, b_id)

        final_cell_array.InsertNextCell(poly_line)
        branch_ids_cell.InsertNextValue(b_id)

    ### Phase 5: Array settings
    final_net = vtk.vtkPolyData()
    final_net.SetPoints(poly_prepared.GetPoints())
    final_net.SetLines(final_cell_array)
    final_net.GetCellData().AddArray(branch_ids_cell)
    # Add individual branch data
    final_net.GetPointData().PassData(poly_prepared.GetPointData())
    final_net.GetPointData().AddArray(branch_ids_point)

    # Create NodeType
    node_type_array = vtk.vtkIntArray()
    node_type_array.SetName("NodeType")
    node_type_array.SetNumberOfTuples(final_net.GetNumberOfPoints())
    for i in range(final_net.GetNumberOfPoints()):
        node_type_array.SetTuple1(i, node_connectivity[i])
    final_net.GetPointData().AddArray(node_type_array)

    # Crete UsageTag field
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