import vtk
import numpy as np
import os
from collections import defaultdict

# --- CONFIGURACIÓN ---
file_vtp = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines/cow_full_final.vtp"
file_out = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines/solver_input.in"

# Parámetros físicos por defecto (ajustar según tu caso)
DENSITY = 1.06       # g/cm^3 (Sangre)
VISCOSITY = 0.04     # poise (g/cm*s)
MAT_E = 4.0e6        # Módulo de Young (dynes/cm^2)
MAT_HU = 0.1         # Espesor relativo de pared (h/R)

# 1. CARGAR DATOS
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(file_vtp)
reader.Update()
poly = reader.GetOutput()

radius_array = poly.GetPointData().GetArray("MaximumInscribedSphereRadius")
node_type_array = poly.GetPointData().GetArray("NodeType")

# 2. MAPEAR NODOS ÚNICOS
# Necesitamos identificar qué puntos físicos son extremos o junctions
node_coords = {}
for i in range(poly.GetNumberOfPoints()):
    if node_type_array.GetTuple1(i) != 2: # 1 (extremo) o 3+ (junction)
        node_coords[i] = poly.GetPoint(i)

# 3. EXTRAER SEGMENTOS (Ramas)
segments = []
joint_map = defaultdict(list) # Para saber qué ramas entran/salen de cada nodo

for i in range(poly.GetNumberOfCells()):
    cell = poly.GetCell(i)
    ids = cell.GetPointIds()
    n_pts = ids.GetNumberOfIds()
    
    id_start = ids.GetId(0)
    id_end = ids.GetId(n_pts - 1)
    
    # Calcular longitud real sumando distancias entre puntos
    length = 0.0
    radii = []
    for j in range(n_pts - 1):
        p1 = np.array(poly.GetPoint(ids.GetId(j)))
        p2 = np.array(poly.GetPoint(ids.GetId(j+1)))
        length += np.linalg.norm(p1 - p2)
        radii.append(radius_array.GetTuple1(ids.GetId(j)))
    radii.append(radius_array.GetTuple1(ids.GetId(n_pts-1)))
    
    avg_radius = np.mean(radii)
    
    seg_name = f"Branch_{i}"
    segments.append({
        'name': seg_name,
        'node_in': id_start,
        'node_out': id_end,
        'L': length,
        'R': avg_radius
    })
    
    # Registrar conectividad para los JOINTS
    joint_map[id_start].append({'name': seg_name, 'type': 'OUT'})
    joint_map[id_end].append({'name': seg_name, 'type': 'IN'})

# 4. ESCRIBIR ARCHIVO .IN PARA SIMVASCULAR
with open(file_out, 'w') as f:
    f.write("# --- SimVascular 1D Solver Input File ---\n\n")
    
    # BLOQUE DE NODOS
    for nid, coords in node_coords.items():
        f.write(f"NODE {nid} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")
    
    f.write("\n")

    # BLOQUE DE SEGMENTOS
    # Formato: SEGMENT name id L num_points node_in node_out R_in R_out mat_id ...
    for i, s in enumerate(segments):
        f.write(f"SEGMENT {s['name']} {i} {s['L']:.4f} 2 {s['node_in']} {s['node_out']} {s['R']:.4f} {s['R']:.4f} 0\n")
    
    f.write("\n")
    
    # BLOQUE DE MATERIAL (ID 0 por defecto)
    f.write(f"MATERIAL MAT0 RIGID {DENSITY} {VISCOSITY}\n\n")

    # BLOQUE DE UNIONES (JOINTS)
    # Solo para puntos donde conectan más de una rama o terminales
    for nid, conns in joint_map.items():
        in_segs = [c['name'] for c in conns if c['type'] == 'IN']
        out_segs = [c['name'] for c in conns if c['type'] == 'OUT']
        
        # En SV 1D, un JOINT conecta entradas con salidas
        f.write(f"JOINT J_{nid} {nid}\n")
        if in_segs: f.write(f"  IN {' '.join(in_segs)}\n")
        if out_segs: f.write(f"  OUT {' '.join(out_segs)}\n")

print(f"Archivo SV 1D generado con éxito en: {file_out}")