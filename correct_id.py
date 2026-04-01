import vtk
from vtk.util import numpy_support
import numpy as np
import os
import pandas as pd


"""
# --- CONFIGURACIÓN ---
og_dir = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models"
input_file = os.path.join(og_dir, "no_collaterals.vtp")
output_file = os.path.join(og_dir,"no_collaterals_reindexed.vtp")
array_name = "ModelFaceID"
escala_factor = 0.1  # Factor para pasar de mm (x10) a cm (x1)

# 1. Leer el archivo
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(input_file)
reader.Update()
polydata = reader.GetOutput()

# --- NUEVA FASE: ESCALADO ---
print(f"--- Aplicando factor de escala: {escala_factor} ---")

# A. Escalar la geometría (puntos X, Y, Z)
transform = vtk.vtkTransform()
transform.Scale(escala_factor, escala_factor, escala_factor)

transformFilter = vtk.vtkTransformPolyDataFilter()
transformFilter.SetInputData(polydata)
transformFilter.SetTransform(transform)
transformFilter.Update()

polydata = transformFilter.GetOutput()

# B. Escalar arrays de Radio (si existen en el PointData)
# VMTK y SimVascular suelen usar estos nombres
nombres_radios = ["Radius", "MaximumInscribedSphereRadius"]
for nombre in nombres_radios:
    radio_array = polydata.GetPointData().GetArray(nombre)
    if radio_array:
        print(f"Escalando valores del array: {nombre}")
        # Convertimos a numpy para operar rápido y devolvemos a VTK
        np_radio = numpy_support.vtk_to_numpy(radio_array)
        np_radio *= escala_factor
        # Actualizamos el array original
        for i in range(len(np_radio)):
            radio_array.SetTuple1(i, np_radio[i])

# --- CONTINUACIÓN: REINDEXACIÓN DE IDs ---

# 2. Obtener el array de IDs original (de la malla ya escalada)
cell_data = polydata.GetCellData().GetArray(array_name)
if not cell_data:
    print(f"Error: No se encontró el array {array_name}")
    exit()

ids_originales = numpy_support.vtk_to_numpy(cell_data)

# 3. Encontrar IDs únicos existentes
unique_ids = sorted(np.unique(ids_originales))
print(f"IDs detectados en el modelo original: {unique_ids}")

# 4. Crear el mapeo: {ID_viejo: ID_nuevo}
mapping = {old_id: i + 1 for i, old_id in enumerate(unique_ids)}

# 5. Crear el nuevo array de IDs
new_ids = np.array([mapping[old_id] for old_id in ids_originales], dtype=np.int32)
new_array = numpy_support.numpy_to_vtk(new_ids, deep=True)
new_array.SetName(array_name)

# 6. Reemplazar el array en el polydata y guardar
polydata.GetCellData().RemoveArray(array_name)
polydata.GetCellData().AddArray(new_array)

# Guardar el archivo final (Escalado + Reindexado)
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(output_file)
writer.SetInputData(polydata)
writer.Write()

print(f"\n¡Proceso completado!")
print(f"Modelo escalado por {escala_factor}")
print(f"IDs reindexados del 1 al {len(unique_ids)} en '{output_file}'")
"""
my_xml = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/326_no_collaterals/Models/no_collaterals.mdl"
df = pd.read_xml(my_xml, xpath=".//face")

face_mapping = dict(zip(df['name'],df['id']))

for key, value in face_mapping.items():
    print(f"{key}: {value}")

cap_names = ["cap_Right_Post", "cap_Right_SCA", "cap_Right_Anterior"]
ids = []
for cap in cap_names:
    ids.append(face_mapping[cap])


print(ids)

