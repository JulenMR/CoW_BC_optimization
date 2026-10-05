# Automatic Boundary Condition parameter setting for Circle of Willis 3D models
### Overview
This pipeline provides optimized Boundary Condition (BC) parameters that match clinical meassurements for Circle of Willis (CoW) patient-specific 3D hemodynamics.
The code is designed to work inside a Simvasclar project file structure and outputs optimized 3-Element Windkessel parameters (Rd, C, Rp) for each CoW outlet matching pressure and flowsplit clinical meassurements.

Sivascular's zeroD solver was chosen as a computationally cheap surrogate model that enables the efficient implementation of optimization algorithms. For this case the L-BFGS optimizer was selected obtaininga fast convergence
with NRMSE <5%. Once the optimum BC parameters are calculated, a stochastic calibration is performed using Sequential Monte Carlo technique. This produces 5% and 95% confidance intervals for each parameter.

The workflow is divided into 3 different phases:
1. Centerline extraction
2. Boundary Condition optimization
3. SMC calibration

### 1. Centerline extraction
Simvascular's default centerline extraction function works by finding the shortest path between the inlets and outlets. However this method fails to capture CoW's looped topology and some branches are left uncovered. Consequently a 
custom centerline extraction function was developed. This function extracts individual centerlines from different inlet/outlet pairs using the VMTK python library and posteriorly merges them into a unified centerline that captures the shape and position of all the branches. 
Additionally this function identifies key points such as:
- Inlets 
- Outlets 
- Junctions
- Stenoses points

### 2. Boundary Condition Optimization
The aim of the second phase is to calculate the optimum BC parameters that match the clinical meassurements by applying the L-BFGS algorithm to 0D surrogates. First of all, it is necessary to translate the centerline into a JSON file
that the ZeroD optimizer understands. To do so, the centerline is interpreted as a Graph network, is explored using the Breadth First Search algorithm and the coordenates of each branch, junction inlet and outlet are included in the 
JSON file that produces the 0D simulation. 

After this, the L-BFGS quasi-Newton optimizer finds the boundary condition parameters that produce simulations with pressures and flow distributions that match clinical meassurements. The obtained 
parameters can be used in a 3D simulation as the difference in results is neglectible. Finally, the .inp file that sets up the 3D simulation using the svFSI solver is automatically generated.

### 3. Uncertainty Quantification
The last steps performs an uncertainty quantification analysis that provides confidance intervals using the Sequential Monte Carlo approach. The values obtained from the L-BFGS optimizer are used as the mean for the prior of each parameter with
a standard deviation of 25%. This process is parallelizable and the number of particles for the process can be selected.

## **How to use this pipeline**
The new implementations to work with multiple sub-branches are in the git branch "multiple_outlet". Once in that branch, entire pipeline can be executed from the `full_process.py` file. 

### Prerequisites & File Structure

1. Clone the repository and select the branch:
```bash
git clone https://github.com/JulenMR/CoW_BC_optimization.git
cd CoW_BC_optimization
git branch -a
git checkout multiple_outlet
``` 
2. Create virtual environment with VMTK library
```bash
conda create -n cow_pipeline_env -c vmtk vmtk python=3.10
conda activate cow_pipeline_env
```
3. Install dependencies with pip:
```bash
pip install -r requirements.txt
```

The script expects the standard SimVascular project structure. Ensure the following files are in the Models/ folder:

* cow_model_coarse.vtp: A lighter coarse remesh of your 3D model (The pipeline works with any model but a coarser version is recommended for faster centerline extraction).

* cow_faceID.mdl: The model face mapping file that matches each caps name with a numeric ID. The cap names must follow the convention "side_outlet" for example "L_SCA" would refer to the left superior cerebellar artery and "R_ACA" to the right anterior communicating artery for example. The new update admits including additional sub-branches for each principal outlet vessel. To refer to those, the naming convention must include an subindex with number "2" in it. For example if the right medium cerebral artery has 2 sub-branches, the names for these must be "R_MCA" and "R_MCA_2".

  
### Input parameters:
**Phase 1**
* *patient number*: Corresponds to the ID number that identifies the patient
* *sv_project_filepath*: The filepath to the Simvascular project root.
* *objective_branches*: This pipeline builds the CoW's global centerline by merging a set of individual centerlines connecting different inlet/outlet combinations. If this input parameter is set to "None" there is a default combination of outlets that captures the geometry of a standard CoW. However, considering CoW's geometrical variability, the option of defining a custom list is allowed for the user. The list has to include a tuple of outlets to generate each individual centerline.
  
objective_branches = [("cap_L_ICA", "cap_L_MCA"), ("cap_R_ICA", "cap_R_MCA")]

* *merging_tolerances*: It is a dictionary that includes 3 tolerance values that the vtk.vtkCleanPolyData() function from the centerline merging phase needs. They specifiy (in mm) the distance threshold to merge 2 close points into one.

  - "General tolerance": Distance threshold to merge two points into one node.
  
  - "ACA tolerance": Specific threshold for Anterior Cerebral Arteries to prevent branch collapse due to anatomical proximity.
  
  - "Spatial tolerance": Maximum distance that a point needs to be from the 3D model's cap to be identified as inlet/outlet.

It is recommended to verify if the obtained "final_centerline.vtp" object correctly captures the model's geometry and if inlets, outlets and junctions are detected successfully. To so so, you can check the "UsageTag" field in paraview.  If the obtained fenterline is not correct, the BC optimization phase will fail.


<img width="320" height="401" alt="image" src="https://github.com/user-attachments/assets/a923e2c7-b62f-4189-8c0e-00f700cdb66d" />

**Phase 2**
* *inflow_filepath*: Path were the inlet flow documents are located. 4 files are expected in the folder, one for each CoW inlet: LICA.dat, RICA.dat, LVA.dat and RVA.dat. These files need to respect the structure suported from Simvascular,
with 2 columns, the first one for the time steps and the second for the flow values in mL/s.
* *clinical_data_file*: A csv file that collects the clinical data from the patient. The clinical_data_file (.csv) must follow the this exact column structure and outlet naming convention shown in the following example (Pressure in mmHg, Flows in mL/s):

| subject | HR   | SBP   | DBP  | VISCOSITY | R_ACA | L_ACA | R_MCA | L_MCA | R_PCA | L_PCA | R_SCA | L_SCA |
|---------|------|-------|------|-----------|-------|-------|-------|-------|-------|-------|-------|-------|
| 1       | 82.3 | 122   | 86   | 0.004     | 0.88  | 1.83  | 2.05  | 0.36  | 0.37  | 0.16  | 0.23  | 0.23  |

If the patient has additional sub-branches new columns with the defined naming convention should be added. For example if there are sub branches in the left posterior cerebral artery and right anterior communicatin artery, the csv should look like: 
| subject | HR   | SBP   | DBP  | VISCOSITY | R_ACA | L_ACA | R_MCA | L_MCA | R_PCA | L_PCA | R_SCA | L_SCA | L_PCA_2 | R_ACA_2 |
|---------|------|-------|------|-----------|-------|-------|-------|-------|-------|-------|-------|-------|---------|-------|
| 1       | 82.3 | 122   | 86   | 0.004     | 0.88  | 1.83  | 2.05  | 0.36  | 0.37  | 0.16  | 0.23  | 0.23  | 0.12    | 0.32  |
  
**Phase 3**
* *num_particles*: Is the number of samples used in the Sequential Monte Carlo (SMC) process.
* *num_cores*: The SMC process can be parallelized. This parameter defines the number of cores used.
* *err_tolerance*: The standard deviation from the normal function that represents the likelihood in the SMC. In other words, it represents the acceptable error percentage between the simulations and clinical data.

### Observations
Units: The pipeline assumes scales in mmgs (mm, g, s).
