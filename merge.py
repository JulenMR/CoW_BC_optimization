import vtk
import os
import glob
from vmtk import vmtkscripts

# --- CONFIGURACIÓN ---
save_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/Centerlines"
output_final = os.path.join(save_file, "cow_final_vmtk.vtp")

files = glob.glob(os.path.join(save_file, "fixed_tmp_cl_*.vtp"))
append_filter = vtk.vtkAppendPolyData()

for i, f in enumerate(files):
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(f)
    r.Update()
    poly = r.GetOutput()
    
    # Creamos el array con el nombre exacto que VMTK reportó en el error: GroupIds
    group_ids = vtk.vtkIntArray()
    group_ids.SetName("GroupIds") # CAMBIO: Nombre exacto del error
    group_ids.SetNumberOfTuples(poly.GetNumberOfCells())
    for j in range(poly.GetNumberOfCells()):
        group_ids.SetTuple1(j, i)
    
    # Lo añadimos tanto a Cells como a Points para evitar fallos de mapeo
    poly.GetCellData().AddArray(group_ids)
    
    # Creamos una versión para puntos (muchos scripts de VMTK la prefieren así)
    point_group_ids = vtk.vtkIntArray()
    point_group_ids.SetName("GroupIds")
    point_group_ids.SetNumberOfTuples(poly.GetNumberOfPoints())
    for j in range(poly.GetNumberOfPoints()):
        point_group_ids.SetTuple1(j, i)
    poly.GetPointData().AddArray(point_group_ids)
    
    append_filter.AddInputData(poly)

append_filter.Update()

# 2. UNIÓN INTELIGENTE
merger = vmtkscripts.vmtkCenterlineMerge()
merger.Centerlines = append_filter.GetOutput()
# Forzamos los nombres de los arrays en el objeto merger
merger.GroupIdArrayName = "GroupIds"
merger.CenterlineIdsArrayName = "GroupIds" 
merger.Execute()

# 3. ATRIBUTOS (Esto genera la topología de red final)
attributes = vmtkscripts.vmtkCenterlineAttributes()
attributes.Centerlines = merger.Centerlines
attributes.Execute()

# 4. GUARDAR
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(output_final)
writer.SetInputData(attributes.Centerlines)
writer.Write()

print(f"\n--- FUSIÓN COMPLETADA ---")
print(f"Archivo: {output_final}")