import vtk
from vtk.util import numpy_support
import numpy as np
import os

# --- CONFIGURACIÓN ---
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models"
input_file = os.path.join(og_dir, "no_collaterals.vtp")
output_file = os.path.join(og_dir,"no_collaterals_reindexed.vtp")
array_name = "ModelFaceID"

# 1. Leer el archivo
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(input_file)
reader.Update()
polydata = reader.GetOutput()

# 2. Obtener el array de IDs original
cell_data = polydata.GetCellData().GetArray(array_name)
ids_originales = numpy_support.vtk_to_numpy(cell_data)

# 3. Encontrar IDs únicos existentes (excluyendo los vacíos como 12 y 21)
# Esto detectará automáticamente qué números tienen celdas asociadas
unique_ids = sorted(np.unique(ids_originales))
print(f"IDs detectados en el modelo original: {unique_ids}")

# 4. Crear el mapeo: {ID_viejo: ID_nuevo}
# Si quieres que empiecen en 1 y lleguen al 23 (si hay 23 reales)
mapping = {old_id: i + 1 for i, old_id in enumerate(unique_ids)}

# 5. Crear el nuevo array de IDs
new_ids = np.array([mapping[old_id] for old_id in ids_originales], dtype=np.int32)
new_array = numpy_support.numpy_to_vtk(new_ids, deep=True)
new_array.SetName(array_name)

# 6. Reemplazar el array en el polydata y guardar
polydata.GetCellData().RemoveArray(array_name)
polydata.GetCellData().AddArray(new_array)

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(output_file)
writer.SetInputData(polydata)
writer.Write()

print(f"\n¡Reindexación completada!")
print(f"El nuevo archivo '{output_file}' ahora tiene IDs del 1 al {len(unique_ids)} sin huecos.")