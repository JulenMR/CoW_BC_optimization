import vtk
import numpy as np
import os
from vtk.util import numpy_support

def generate_1d_input(vtp_path, output_path):
    if not os.path.exists(vtp_path):
        print(f"Error: No se encuentra {vtp_path}")
        return

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()

    # 1. Cargar Arrays
    branch_ids = numpy_support.vtk_to_numpy(polydata.GetCellData().GetArray("BranchID"))
    usage_tags = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("UsageTag"))
    radii = numpy_support.vtk_to_numpy(polydata.GetPointData().GetArray("MaximumInscribedSphereRadius"))

    # Estructuras para NODES
    node_coords = {} 
    pos_to_node = {} 
    next_node_id = 0

    def get_node(pt_id):
        nonlocal next_node_id
        if pt_id is None: return None
        coords = polydata.GetPoint(pt_id)
        key = tuple(np.round(coords, 6))
        if key not in pos_to_node:
            pos_to_node[key] = next_node_id
            node_coords[next_node_id] = coords
            next_node_id += 1
        return pos_to_node[key]

    print("-" * 60)
    print("Registrando puntos críticos...")
    for i in range(polydata.GetNumberOfPoints()):
        tag = int(usage_tags[i])
        if tag != 0:
            get_node(i)
    print("-" * 60)

    # 2. Procesar cada Branch
    segments = {}
    print(f"\n{'BRANCH':<8} | {'N_IN':<5} | {'N_OUT':<5} | {'R_IN':<8} | {'R_OUT':<8} | {'INFO'}")
    print("-" * 75)

    for i in range(polydata.GetNumberOfCells()):
        cell = polydata.GetCell(i)
        b_id = int(branch_ids[i])
        num_pts = cell.GetNumberOfPoints()
        
        branch_point_ids = [cell.GetPointId(j) for j in range(num_pts)]
        tags_in_branch = [int(usage_tags[pid]) for pid in branch_point_ids]

        p_idx_inlet = next((pid for pid in branch_point_ids if usage_tags[pid] == 1), None)
        p_idx_outlet = next((pid for pid in branch_point_ids if usage_tags[pid] == 2), None)

        # Extremos originales de la celda
        p_first = branch_point_ids[0]
        p_last = branch_point_ids[-1]

        # --- LÓGICA DE DIRECCIÓN Y ASIGNACIÓN DE RADIOS ---
        if p_idx_inlet is not None:
            # Entrada forzada por TAG 1
            idx_in = p_idx_inlet
            idx_out = p_last if idx_in != p_last else p_first
            u_type = 1
            info = "Forced Inlet"
        elif p_idx_outlet is not None:
            # Salida forzada por TAG 2
            idx_out = p_idx_outlet
            idx_in = p_first if idx_out != p_first else p_last
            u_type = 2
            info = "Forced Outlet"
        else:
            # Rama interna estándar
            idx_in = p_first
            idx_out = p_last
            u_type = 3
            info = "Internal"

        n_in = get_node(idx_in)
        n_out = get_node(idx_out)
        
        # ASIGNACIÓN CORRECTA DE RADIOS basada en los puntos seleccionados
        r_in = float(radii[idx_in])
        r_out = float(radii[idx_out])

        if n_in == n_out: info += " [!] CIRCULAR"

        # Cálculo de longitud
        pts = [np.array(polydata.GetPoint(pid)) for pid in branch_point_ids]
        length = sum(np.linalg.norm(pts[j+1] - pts[j]) for j in range(num_pts-1))

        print(f"{b_id:<8} | {n_in:<5} | {n_out:<5} | {r_in:<8.4f} | {r_out:<8.4f} | {info}")

        segments[b_id] = {
            'nodes': (n_in, n_out),
            'length': length,
            'r_in': r_in,
            'r_out': r_out,
            'usage': u_type
        }
    print("-" * 75)

    # 3. Reordenar IDs
    try:
        inlet_bid = next(bid for bid, d in segments.items() if d['usage'] == 1)
    except StopIteration:
        inlet_bid = list(segments.keys())[0]

    ordered_ids = {inlet_bid: 0}
    curr = 1
    for bid in segments:
        if bid != inlet_bid:
            ordered_ids[bid] = curr
            curr += 1

    # 4. Escritura
    with open(output_path, 'w') as f:
        f.write("MODEL ArterialNetwork\n\n")

        for n_id, c in sorted(node_coords.items()):
            f.write(f"NODE {n_id} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        
        f.write("\n")

        for n_id in node_coords:
            in_segs = [ordered_ids[b] for b, d in segments.items() if d['nodes'][1] == n_id]
            out_segs = [ordered_ids[b] for b, d in segments.items() if d['nodes'][0] == n_id]
            if in_segs and out_segs:
                f.write(f"JOINT J{n_id} {n_id} INSEGS OUTSEGS\n")
                f.write(f"JOINTINLET INSEGS {len(in_segs)} {' '.join(map(str, in_segs))}\n")
                f.write(f"JOINTOUTLET OUTSEGS {len(out_segs)} {' '.join(map(str, out_segs))}\n\n")

        for bid, sv_id in sorted(ordered_ids.items(), key=lambda x: x[1]):
            s = segments[bid]
            bc = "RESISTANCE R_VALS" if s['usage'] == 2 else "NOBOUND NONE"
            f.write(f"SEGMENT seg{sv_id} {sv_id} {s['length']:.4f} 50 {s['nodes'][0]} {s['nodes'][1]} "
                    f"{s['r_in']:.4f} {s['r_out']:.4f} 0.0 MAT1 NONE 0.0 0 0 {bc}\n")

        f.write("\nDATATABLE R_VALS LIST\n0.0 1000.0\nENDDATATABLE\n")
        f.write("\nDATATABLE STEADY_FLOW LIST\n0.0 0.0\n0.1 0.5\n1.0 0.5\nENDDATATABLE\n")
        f.write("\nMATERIAL MAT1 OLUFSEN 1.06 0.04 0.0 2.0 1.0e9 -20 1e9\n")
        
        inlet_node = segments[inlet_bid]['nodes'][0]
        f.write(f"\nSOLVEROPTIONS 0.0001 100 500 2 STEADY_FLOW FLOW 1.0e-6 1 1\n")
        f.write("\nOUTPUT VTK\n")

    print(f"\nArchivo generado con éxito: {output_path}")

# Ejecución
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines/medium_example"
centerline_path = os.path.join(og_dir, "cow_medium_final.vtp")
save_path = os.path.join(og_dir, "final_script.in")

generate_1d_input(centerline_path, save_path)