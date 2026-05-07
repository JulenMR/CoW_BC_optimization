import numpy as np
import particles
from particles import distributions as dists
from particles import state_space_models as ssm 
import json
import pysvzerod
import copy
from generate_json import get_clinical_data
from particles import collectors
import multiprocessing as mp
from scipy import stats  
import corner
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time 

class VLikelihood(dists.ProbDist):
    def __init__(self, log_val):
        # log_val ahora será un array de tamaño N
        self.log_val = np.atleast_1d(log_val)
        self.dim = 1 # Dimensión del dato observado
        

    def logpdf(self, x):
        # x es el dato observado (data[t]), devolvemos el log_val guardado
        return self.log_val
    

class RCR_UQ(ssm.StateSpaceModel):
    def __init__(self, json_dict, active_rcr_ids, clinical_targets, branch_map, mapping_dict, **kwargs):
        super().__init__(**kwargs)
        self.json_base = json_dict
        self.active_ids = active_rcr_ids
        self.targets = clinical_targets 
        self.branch_map = branch_map
        self.mapping_dict = mapping_dict
        self.num_outlets = len(active_rcr_ids)
        self.history = []

    def PX0(self): # Prior
        priors = []
        for _ in range(self.num_outlets):
            priors.extend([
                dists.Uniform(0.1, 10.0),   # Rp
                dists.Uniform(0.001, 0.8),  # C
                dists.Uniform(0.5, 50.0)    # Rd
            ])
        return dists.IndepProd(*priors)

    def PX(self, t, xp): # Constant parameters for Jitter
        return dists.Normal(loc=xp, scale=1e-2)

    def PY(self, t, xp, x): # Likelihood
       
        N = x.shape[0] if len(x.shape) > 1 else 1
        log_weights = np.zeros(N)
        
        if N == 1:
            log_weights[0] = self._evaluate_single_particle(x)
        else:

            for i in range(N):
                log_weights[i] = self._evaluate_single_particle(x[i])
        
        return VLikelihood(log_weights)
    
    def _evaluate_single_particle(self, theta):
        js = copy.deepcopy(self.json_base)
        update_all_outlets(js, theta, self.active_ids)
        try:
            solver = pysvzerod.Solver(js)
            solver.run()
            df = solver.get_full_result()
            sim_mp, sim_pulse, sim_f = get_stats(df, self.active_ids)
            record = {
                'theta': theta.copy(),
                'p_mean': sim_mp,
                'p_pulse': sim_pulse,
                'flows': list(sim_f.values())
            }
            self.history.append(record)
            
            y_sim = [sim_f[f"branch{idx}"] for idx in self.active_ids if f"branch{idx}" in sim_f]
            y_sim.append(sim_mp)
            y_sim.append(sim_pulse)
            
            y_target = [self.targets['flows'][name] for name in self.branch_map if f"branch{self.mapping_dict[name]}" in sim_f]
            y_target.append(self.targets['mean_p'])
            y_target.append(self.targets["pulse"])
            
            scales = [max(0.1, 0.1 * val) for val in y_target[:-2]] + [0.5, 0.1] 
    
            # 3. Calcular logprobs
            log_probs = stats.norm.logpdf(y_target, loc=y_sim, scale=scales)
            
            # 4. BOOSTING: El pulso es 10 veces más importante que el resto
            # Esto "estira" la montaña de probabilidad para que sea un pico afilado
            log_probs[-1] *= 10 

            return np.sum(log_probs)
        except:
            return -1e10

    
def update_all_outlets(json_dict, x, active_rcr_ids):
    bc_list = json_dict["boundary_conditions"]
    for i, rcr_id in enumerate(active_rcr_ids):
        idx = i * 3
        target_bc_name = f"RCR_{rcr_id}"
        
        for bc in bc_list:
            if bc.get("bc_name") == target_bc_name:
                bc["bc_values"]["Rp"] = x[idx]
                bc["bc_values"]["C"]  = x[idx + 1]
                bc["bc_values"]["Rd"] = x[idx + 2]

def get_stats(df, active_rcr_ids):
    inlet_names = ["branch0", "branch1", "branch2", "branch3"]
    p_means, p_pulses = [], []
    
    available_branches = df['name'].unique()
    for name in inlet_names:
        if name in available_branches:
            data = df[df['name'] == name]
            p_vals = data["pressure_in"].values / 133.32
            p_means.append(np.mean(p_vals))
            p_pulses.append(np.max(p_vals) - np.min(p_vals))
    
    outlet_stats = {}
    for rcr_id in active_rcr_ids:
        name = f"branch{rcr_id}"
        if name in available_branches:
            data = df[df['name'] == name]
            outlet_stats[name] = np.mean(data["flow_out"].values)
        
    return np.mean(p_means), np.mean(p_pulses), outlet_stats

def result_visualization(result_npy, active_rcr_ids):
    samples = np.load(result_npy)

    # 1. Corner Plot (Se mantiene igual, está perfecto)
    params_outlet = samples[:, 0:3] 
    labels = ["$R_p$", "$C$", "$R_d$"]
    fig = corner.corner(
        params_outlet, 
        labels=labels,
        quantiles=[0.16, 0.5, 0.84], 
        show_titles=True, 
        title_kwargs={"fontsize": 12}
    )
    plt.show()

    results_table = []
    for i, rcr_id in enumerate(active_rcr_ids):
        idx = i * 3
        res = {
            "Outlet": rcr_id,
            "Rp_mean": np.mean(samples[:, idx]),
            "Rp_std":  np.std(samples[:, idx]),
            "C_mean":  np.mean(samples[:, idx+1]),
            "C_std":   np.std(samples[:, idx+1]),
            "Rd_mean": np.mean(samples[:, idx+2]),
            "Rd_std":  np.std(samples[:, idx+2])
        }
        results_table.append(res)

    df_results = pd.DataFrame(results_table)
    print(df_results)
    #df_results.to_csv("summary_results_SMC.csv", index=False)

def plot_all_parameters(result_npy, active_rcr_ids):
    samples = np.load(result_npy)
    num_outlets = len(active_rcr_ids)
    
    # Creamos una figura grande: 3 filas (Rp, C, Rd) x N columnas (Outlets)
    fig, axes = plt.subplots(3, num_outlets, figsize=(4 * num_outlets, 10), sharey=False)
    
    param_names = ["Rp", "C", "Rd"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"] # Azul, Rojo, Verde

    for row, p_name in enumerate(param_names):
        for col, rcr_id in enumerate(active_rcr_ids):
            ax = axes[row, col]
            
            # Índice de la columna en el array samples
            # Si row=0 (Rp) -> idx = col*3 + 0
            # Si row=1 (C)  -> idx = col*3 + 1
            # Si row=2 (Rd) -> idx = col*3 + 2
            idx = col * 3 + row
            
            sns.kdeplot(samples[:, idx], ax=ax, fill=True, color=colors[row])
            
            # Solo ponemos el título del outlet en la primera fila
            if row == 0:
                ax.set_title(f"Outlet {rcr_id}", fontsize=14, fontweight='bold')
            
            # Solo ponemos el nombre del parámetro en la primera columna
            if col == 0:
                ax.set_ylabel(p_name, fontsize=14, fontweight='bold')
            else:
                ax.set_ylabel("")

            ax.tick_params(axis='x', rotation=45)

    plt.suptitle("Distribuciones Posteriores por Parámetro y Outlet", fontsize=20, y=1.02)
    plt.tight_layout()
    #plt.savefig("full_parameter_distribution.png", bbox_inches='tight')
    plt.show()

def diagnostic_random_particles(alg, rcr_model, n_samples=3):
    """
    Toma n partículas aleatorias del resultado final y muestra 
    su desempeño real frente a los targets.
    """
    print(f"\n{'#'*60}")
    print(f"{'# DIAGNÓSTICO DE PARTÍCULAS ALEATORIAS (POSTERIOR)':^58} #")
    print(f"{'#'*60}")

    # Seleccionar índices aleatorios
    indices = np.random.choice(len(alg.X), n_samples, replace=False)
    
    for idx in indices:
        theta = alg.X[idx]
        
        # Ejecutar simulación física
        js = copy.deepcopy(rcr_model.json_base)
        update_all_outlets(js, theta, rcr_model.active_ids)
        
        solver = pysvzerod.Solver(js)
        solver.run()
        df = solver.get_full_result()
        
        sim_mp, sim_pulse, sim_f = get_stats(df, rcr_model.active_ids)
        
        # Preparar datos para la tabla (mapeando nombres de vasos)
        print(f"\n>>> PARTÍCULA INDEX: {idx}")
        print(f"{'='*55}")
        print(f" Mean pressure: {sim_mp:5.1f} (Target: {rcr_model.targets['mean_p']:5.1f})")
        print(f" Pulse:         {sim_pulse:5.1f} (Target: {rcr_model.targets['pulse']:5.1f})")
        print(f"{'-'*55}")
        print(f"{'Vessel':<12} | {'Simulated':>12} | {'Target':>12} | {'Error %':>8}")
        print(f"{'-'*55}")

        total_err = 0
        for name, branch_idx in rcr_model.branch_map.items():
            b_key = f"branch{branch_idx}"
            if b_key in sim_f:
                sim = sim_f[b_key]
                target = rcr_model.targets['flows'][name]
                err_pct = abs(sim - target) / target * 100
                total_err += err_pct
                print(f"{name:<12} | {sim:12.2f} | {target:12.2f} | {err_pct:7.1f}%")
        
        print(f"{'-'*55}")
        print(f"Mean Flow Error: {total_err/len(sim_f):5.2f}%")
        print(f"{'='*55}\n")

if __name__ == "__main__":
    start_time = time.time()
    patient_number = 5
    clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/subject_targets.csv"
    json_path = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-005/Models/zeroD_simulation/zeroD_script_optimized.json"
    mapping_dict = {
        "L_ICA":0, "R_ICA":1, "L_VA":2, "R_VA":3, "L_SCA":4, "L_PCA":5,
        "L_MCA":6, "L_ACA":7, "R_ACA":8, "R_MCA":9, "R_PCA":10, "R_SCA":11,
    }

    with open(json_path, 'r') as f: json_dict = json.load(f)
    active_rcr_ids = []
    for i in range(4, 12):
        bc_list = json_dict["boundary_conditions"]
        branch_count = 0
        try:
            params = next(bc["bc_values"] for bc in bc_list if bc.get("bc_name") == f"RCR_{i}")
            active_rcr_ids.append(i)
        except StopIteration:
            continue

    branch_count = len(active_rcr_ids)
    print(f"Active outlets found: {branch_count} ({active_rcr_ids})")

    mean_p, pulse, clinical_flows  = get_clinical_data(file=clinical_data_file, p_number=patient_number)
    targets = {"mean_p": mean_p,
               "pulse": pulse,
               "flows": clinical_flows}
    
    rcr_model = RCR_UQ(json_dict=json_dict, 
                       active_rcr_ids=active_rcr_ids, 
                       clinical_targets=targets, 
                       branch_map=mapping_dict, 
                       mapping_dict=mapping_dict)

    fk_boot = ssm.Bootstrap(ssm=rcr_model, data=np.zeros(1))

    N_particles = 200
    
    print(f"Executing multiSMC in parallel...")

    results = particles.multiSMC(fk=fk_boot, 
                                 N=N_particles, 
                                 nruns=1, 
                                 nprocs=16, 
                                 out_func=None)


    alg = results[0]['output']
    diagnostic_random_particles(alg, rcr_model, n_samples=3)

    np.save(f"particles_patient_{patient_number}.npy", alg.X)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    
    print(f"SMC was successfull. {alg.X.shape[0]} samples for {alg.X.shape[1]} parameters have been created.")

    result_visualization("particles_patient_5.npy", active_rcr_ids)
    plot_all_parameters("particles_patient_5.npy", active_rcr_ids)

    

