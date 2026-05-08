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
        self.log_val = np.atleast_1d(log_val)
        self.dim = 1 
    
    def logpdf(self, x):
        return self.log_val
    

class RCR_UQ(ssm.StateSpaceModel):
    def __init__(self, json_dict, active_rcr_ids, clinical_targets, branch_map, lbfgs_vals, **kwargs):
        super().__init__(**kwargs)
        self.json_base = json_dict
        self.active_ids = active_rcr_ids
        self.targets = clinical_targets 
        self.branch_map = branch_map
        self.num_outlets = len(active_rcr_ids)
        self.lbfgs_vals = lbfgs_vals

    def PX0(self): # Prior
        priors = []
        for val in self.lbfgs_vals: # Prior is centered in the optimum value from lbfgs
            priors.append(dists.LogNormal(mu=np.log(val), sigma=0.25))
        return dists.IndepProd(*priors)

    def PX(self, t, xp): # Constant parameters for Jitter
        return dists.Normal(loc=xp, scale=xp * 0.02)

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
            rel_errors = []
        
            for i, idx in enumerate(self.active_ids):
                target = self.targets['flows'][list(self.branch_map.keys())[i]]
                sim = sim_f[f"branch{idx}"]
                rel_errors.append((sim - target) / target)
                
            # Pressure and pulse
            rel_errors.append((sim_mp - self.targets['mean_p']) / self.targets['mean_p'])
            rel_errors.append((sim_pulse - self.targets['pulse']) / self.targets['pulse'])
            
            # Compare error distribution with a normal with mean 0 and std 0.05
            log_liks = stats.norm.logpdf(rel_errors, loc=0, scale=0.05)
            
            # Extra weight to match pressure
            log_liks[-2:] *= 15 
            
            return np.sum(log_liks)
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

def plot_corner_per_outlet(result_npy, active_rcr_ids, mapping_dict):
    samples = np.load(result_npy)
    id_to_name = {v: k for k, v in mapping_dict.items()}
    labels = ["$R_p$", "$C$", "$R_d$"]
    
    for i, rcr_id in enumerate(active_rcr_ids):
        # El índice en samples siempre será i*3
        idx_start = i * 3
        idx_end = idx_start + 3
        params_subset = samples[:, idx_start:idx_end]
        
        vessel_name = id_to_name.get(rcr_id, f"ID_{rcr_id}")
        
        fig = corner.corner(
            params_subset, 
            labels=labels,
            quantiles=[0.16, 0.5, 0.84], 
            show_titles=True, 
            title_kwargs={"fontsize": 12}
        )
        
        fig.subplots_adjust(top=0.9) 
        fig.suptitle(f"POSTERIOR DISTRIBUTION: {vessel_name}", 
                     fontsize=16, 
                     fontweight='bold',
                     y=0.98)
        
        plt.show()

def plot_all_parameters(result_npy, active_rcr_ids, mapping_dict):
    samples = np.load(result_npy)
    num_outlets = len(active_rcr_ids)
    
    id_to_name = {v: k for k, v in mapping_dict.items()}
    
    fig, axes = plt.subplots(3, num_outlets, figsize=(3 * num_outlets + 4, 10), sharey=False)
    param_names = ["$R_p$", "$C$", "$R_d$"]
    colors = ["blue", "red", "green"] 

    for row, p_name in enumerate(param_names):
        for col, rcr_id in enumerate(active_rcr_ids):
            ax = axes[row, col]
            idx = col * 3 + row
            
            sns.kdeplot(samples[:, idx], ax=ax, fill=True, color=colors[row])
            
            if row == 0:
                vessel_name = id_to_name.get(rcr_id, f"ID {rcr_id}")
                ax.set_title(vessel_name, fontsize=15, fontweight='bold', pad=15)
            
            if col == 0:
                ax.set_ylabel(p_name, fontsize=16, fontweight='bold', labelpad=20)
            else:
                ax.set_ylabel("")

            ax.tick_params(axis='x', rotation=30, labelsize=10)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4)) 

    plt.suptitle("Posteriors for all parameters", 
                 fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Ajuste fino final para los nombres de los parámetros en la izquierda
    plt.subplots_adjust(left=0.1, wspace=0.3, hspace=0.5) 
    
    plt.show()


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
    deterministic_param_values = []
    for i in range(4, 12):
        bc_list = json_dict["boundary_conditions"]
        branch_count = 0
        try:
            params = next(bc["bc_values"] for bc in bc_list if bc.get("bc_name") == f"RCR_{i}")
            for n, val in enumerate(params.values()):
                if n < 3:
                  deterministic_param_values.append(round(val,6))
            active_rcr_ids.append(i)
        except StopIteration:
            continue

    branch_count = len(active_rcr_ids)
    print(f"Active outlets found: {branch_count} ({active_rcr_ids})")
    print(f"parameter list: {deterministic_param_values}")

    mean_p, pulse, clinical_flows  = get_clinical_data(file=clinical_data_file, p_number=patient_number)
    targets = {"mean_p": mean_p,
               "pulse": pulse,
               "flows": clinical_flows}
    
    # rcr_model = RCR_UQ(json_dict=json_dict, 
    #                    active_rcr_ids=active_rcr_ids, 
    #                    clinical_targets=targets, 
    #                    branch_map=mapping_dict, 
    #                    lbfgs_vals = deterministic_param_values)

    # fk_boot = ssm.Bootstrap(ssm=rcr_model, data=np.zeros(1))

    # N_particles = 2000
    
    # print(f"Executing multiSMC in parallel...")

    # results = particles.multiSMC(fk=fk_boot, 
    #                              N=N_particles, 
    #                              nruns=1, 
    #                              nprocs=16, 
    #                              out_func=None)


    # alg = results[0]['output']
    # diagnostic_random_particles(alg, rcr_model, n_samples=3)

    # np.save(f"particles_patient_{patient_number}.npy", alg.X)
    # end_time = time.time()
    # print(f"Execution time: {end_time - start_time} seconds")
    
    # print(f"SMC was successfull. {alg.X.shape[0]} samples for {alg.X.shape[1]} parameters have been created.")

#plot_corner_per_outlet("particles_patient_5.npy", active_rcr_ids, mapping_dict)
plot_all_parameters("particles_patient_5.npy", active_rcr_ids, mapping_dict)

    

