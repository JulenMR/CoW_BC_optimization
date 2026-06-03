import numpy as np
import particles
from particles import distributions as dists
from particles import state_space_models as ssm 
import json
import pysvzerod
import copy
from generate_json import get_clinical_data
import multiprocessing as mp
from scipy import stats  
import corner
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time 
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import os
import scipy.stats as stats
from joblib import Parallel, delayed
from create_3dsim_file import update_svfsi
import matplotlib
matplotlib.use('Agg')
class VLikelihood(dists.ProbDist):
    def __init__(self, log_val):
        self.log_val = np.atleast_1d(log_val)
        self.dim = 1 
    
    def logpdf(self, x):
        return self.log_val

progress_counter = None

def init_pool(counter):
    global progress_counter
    progress_counter = counter

class RCR_UQ(ssm.StateSpaceModel):
    def __init__(self, json_dict, active_rcr_ids, clinical_targets, branch_map, lbfgs_vals, error_tolerance, num_cores, **kwargs):
        super().__init__(**kwargs)
        self.json_base = json_dict
        self.active_ids = active_rcr_ids
        self.targets = clinical_targets 
        self.branch_map = branch_map
        self.num_outlets = len(active_rcr_ids)
        self.lbfgs_vals = lbfgs_vals
        self.error_tolerance = error_tolerance
        self.num_cores = num_cores

    def PX0(self): # Prior
        priors = []
        for val in self.lbfgs_vals: # Prior is centered in the optimum value from lbfgs
            priors.append(dists.LogNormal(mu=np.log(val), sigma=0.25))
        return dists.IndepProd(*priors)
    
    def PX(self, t, xp): # Constant parameters for Jitter
        return dists.Normal(loc=xp, scale=xp * 0.02)

    def PY(self, t, xp, x): # Likelihood Paralelizada
        N = x.shape[0] if len(x.shape) > 1 else 1
        
        if N == 1:
            log_weights = np.array([self._evaluate_single_particle(x, 0)])
        else:
            print(f"\n[SMC Paso {t}] Evaluando {N} partículas SECUENCIALMENTE...")

            log_weights = Parallel(n_jobs=1)(
                delayed(self._evaluate_single_particle)(x[i], i) for i in range(N)
            )
            log_weights = np.array(log_weights)

        os.makedirs("SMC_results", exist_ok=True)
        np.save("SMC_results/temp_raw_particles.npy", x)
        np.save("SMC_results/temp_raw_scores.npy", log_weights)
        
        return VLikelihood(log_weights)
    
    def _evaluate_single_particle(self, theta, particle_idx=None):
        js = copy.deepcopy(self.json_base)
        update_all_outlets(js, theta, self.active_ids)
        
        try:
            solver = pysvzerod.Solver(js)
            solver.run()
            df = solver.get_full_result()
            sim_mp, sim_pulse, sim_f = get_stats(df, self.active_ids)
            
            # 1. Creamos un diccionario inverso: mapea el ID numérico al nombre de la arteria
            # Ejemplo: {0: 'L_ICA', 1: 'R_ICA', 2: 'L_VA', ...}
            id_to_branch = {v: k for k, v in self.branch_map.items()}
            
            rel_errors = []
            
            # 2. Iteramos directamente sobre las arterias activas (las de salida)
            for idx in self.active_ids:
                # Buscamos el nombre correcto usando el ID real
                branch_name = id_to_branch[idx] 
                
                # Extraemos el objetivo clínico usando ese nombre exacto
                target = self.targets['flows'][branch_name]
                
                sim = sim_f[f"branch{idx}"]
                rel_errors.append((sim - target) / target)
                
            rel_errors.append((sim_mp - self.targets['mean_p']) / self.targets['mean_p'])
            rel_errors.append((sim_pulse - self.targets['pulse']) / self.targets['pulse'])
            
            log_liks = stats.norm.logpdf(rel_errors, loc=0, scale=self.error_tolerance)
            log_liks[-2:] *= 2 
            final_score = np.sum(log_liks)
            
            if particle_idx is not None and particle_idx % 20 == 0:
                active_branches = [id_to_branch[uid] for uid in self.active_ids]
                self._generate_debug_plot(particle_idx, active_branches, sim_f, sim_mp, sim_pulse, final_score)
                # Añadimos flush=True para forzar la impresión inmediata
                print(f"✅ Partícula {particle_idx} OK. Score: {final_score:.2f} (Gráfico guardado)", flush=True)
            
            return final_score
            
        except Exception as e:
            if particle_idx is not None and particle_idx % 20 == 0:
                # Añadimos flush=True aquí también
                print(f"❌ Partícula {particle_idx} RECHAZADA. Error: {e}", flush=True)
                os.makedirs("SMC_debug_plots", exist_ok=True)
                with open(f"SMC_debug_plots/particula_{particle_idx}_RECHAZADA.txt", "w") as f:
                    f.write(f"Estado: RECHAZADA\nError: {e}\n")
            return -1e10

    def _generate_debug_plot(self, idx, branches, sim_f, sim_mp, sim_pulse, score):
        """Genera un reporte gráfico comparando la simulación con los datos clínicos."""
        os.makedirs("SMC_debug_plots", exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Partícula Index {idx} - ACEPTADA (Score/LogLik: {score:.2f})", fontsize=14, fontweight='bold')
        
        # 1. Gráfico de Flujos (Simulado vs Clínico)
        sim_flows = [sim_f[f"branch{self.active_ids[i]}"] for i in range(len(self.active_ids))]
        target_flows = [self.targets['flows'][b] for b in branches]
        
        x = np.arange(len(branches))
        width = 0.35
        
        axes[0].bar(x - width/2, target_flows, width, label='Objetivo Clínico', color='#1f77b4')
        axes[0].bar(x + width/2, sim_flows, width, label='Simulación 0D', color='#ff7f0e')
        axes[0].set_ylabel('Flujo')
        axes[0].set_title('Comparación de Flujos por Rama')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(branches, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. Gráfico de Presiones (Media y Pulso)
        labels_p = ['Presión Media', 'Presión Pulso']
        targets_p = [self.targets['mean_p'], self.targets['pulse']]
        sims_p = [sim_mp, sim_pulse]
        
        x_p = np.arange(len(labels_p))
        
        axes[1].bar(x_p - width/2, targets_p, width, label='Objetivo Clínico', color='#1f77b4')
        axes[1].bar(x_p + width/2, sims_p, width, label='Simulación 0D', color='#ff7f0e')
        axes[1].set_ylabel('Presión (mmHg o equivalente)')
        axes[1].set_title('Presión Arterial Global')
        axes[1].set_xticks(x_p)
        axes[1].set_xticklabels(labels_p)
        axes[1].legend()
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"SMC_debug_plots/particula_{idx}_ACEPTADA.png", dpi=150)
        plt.close()

    
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

def plot_all_parameters(sv_project_file, result_npy, active_rcr_ids, mapping_dict, patient_num, save= False):
    samples = np.load(result_npy)
    num_outlets = len(active_rcr_ids)
    
    id_to_name = {v: k for k, v in mapping_dict.items()}
    
    fig, axes = plt.subplots(3, num_outlets, figsize=(4 * num_outlets, 12), sharey=False)
    param_names = ["$R_p$", "$C$", "$R_d$"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"] 

    for row, p_name in enumerate(param_names):
        for col, rcr_id in enumerate(active_rcr_ids):
            ax = axes[row, col]
            idx = col * 3 + row
            data = samples[:, idx]
            
            p5 = np.percentile(data, 5)
            p95 = np.percentile(data, 95)
            media = np.mean(data)
            
            # Dibujar KDE
            sns.kdeplot(data, ax=ax, fill=True, color=colors[row], alpha=0.4)
            
            # Líneas verticales
            ax.axvline(p5, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
            ax.axvline(p95, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
            ax.axvline(media, color='black', linestyle='-', linewidth=1.5)

            ticks = [p5, media, p95]
            ax.set_xticks(ticks)
            
            labels = [f"{p5:.3f}", f"{media:.3f}", f"{p95:.3f}"]
            ax.set_xticklabels(labels, rotation=45, fontsize=9)

            if row == 0:
                vessel_name = id_to_name.get(rcr_id, f"ID {rcr_id}")
                ax.set_title(vessel_name, fontsize=15, fontweight='bold', pad=20)
            
            if col == 0:
                ax.set_ylabel(p_name, fontsize=16, fontweight='bold', labelpad=20)
            else:
                ax.set_ylabel("")

            ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.suptitle(f"Posterior distributions for PACS{patient_num:03d}", 
                 fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(left=0.08, wspace=0.4, hspace=0.6) 
    if save:
        save_path = os.path.join(sv_project_file, "ROMSimulations", f"uq_pacs{patient_num:03d}.png")
        plt.savefig(save_path)
        print(f"Posterior distribution image saved in {save_path}")
    plt.close()

def _generate_debug_plot_custom(idx, branches, sim_f, sim_mp, sim_pulse, score, folder, targets, active_rcr_ids):
    """Versión ajustada para guardar en carpetas específicas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Validación Partícula {idx}", fontsize=14)
    
    # Gráficos de Flujos
    sim_flows = [sim_f[f"branch{active_rcr_ids[i]}"] for i in range(len(active_rcr_ids))]
    target_flows = [targets['flows'][b] for b in branches]
    x = np.arange(len(branches))
    
    axes[0].bar(x - 0.15, target_flows, 0.3, label='Clínico')
    axes[0].bar(x + 0.15, sim_flows, 0.3, label='Simulado')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(branches, rotation=45)
    axes[0].legend()
    
    # Gráficos de Presión
    axes[1].bar([0, 1], [targets['mean_p'], targets['pulse']], 0.3, label='Clínico')
    axes[1].bar([0.3, 1.3], [sim_mp, sim_pulse], 0.3, label='Simulado')
    axes[1].set_xticks([0.15, 1.15])
    axes[1].set_xticklabels(['Presión Media', 'Presión Pulso'])
    axes[1].legend()
    
    plt.savefig(f"{folder}/sample_{idx}.png")
    plt.close()

def validate_confidence_region(particles_npy, scores_npy, json_path, active_rcr_ids, clinical_targets, branch_map, n_samples=5):
    with open(json_path, 'r') as f: json_base = json.load(f)
    # 1. Cargar datos
    particles = np.load(particles_npy)
    scores = np.load(scores_npy)
    
    # 2. Calcular el centroide (media de los parámetros)
    # Esto define el "punto central" de tu región de confianza
    media_particulas = np.mean(particles, axis=0)
    
    # 3. Calcular distancia euclidiana de cada partícula al centro
    # Usamos la norma L2 para saber qué tan "típica" es cada partícula
    distancias = np.linalg.norm(particles - media_particulas, axis=1)
    
    # 4. Seleccionar partículas dentro del 5-95% de confianza
    # Ordenamos por distancia: las primeras son las más cercanas a la "media"
    indices_cercanos = np.argsort(distancias)
    
    # Tomamos n_samples partículas que representan el centro de la distribución
    # (aquí descartamos las que tienen distancias muy altas, que son tus outliers)
    indices_seleccionados = indices_cercanos[:n_samples]
    
    muestras_confianza = particles[indices_seleccionados]
    scores_confianza = scores[indices_seleccionados]
    
    # 5. Ejecutar y visualizar
    os.makedirs("sample_results", exist_ok=True)
    id_to_branch = {v: k for k, v in branch_map.items()}
    
    print(f"Validando {n_samples} partículas representativas de la región de confianza...")
    
    for i, (theta, score) in enumerate(zip(muestras_confianza, scores_confianza)):
        js = copy.deepcopy(json_base)
        update_all_outlets(js, theta, active_rcr_ids)
        
        # Correr solver
        solver = pysvzerod.Solver(js)
        solver.run()
        df = solver.get_full_result()
        sim_mp, sim_pulse, sim_f = get_stats(df, active_rcr_ids)
        
        # Guardar plot
        _generate_debug_plot_custom(
            idx=i, 
            branches=[id_to_branch[uid] for uid in active_rcr_ids], 
            sim_f=sim_f, sim_mp=sim_mp, sim_pulse=sim_pulse, 
            score=score, folder="sample_results", 
            targets=clinical_targets, active_rcr_ids=active_rcr_ids
        )
        inp_path = "./svFSI_base.inp"
        save_path = f"sample_results/sample_{i}.inp"
        update_svfsi(json_path, inp_path, mapping_dict, timestep_size=None, save_path=save_path)
        print(f"Partícula {i} visualizada. Score: {score:.2f}")


if __name__ == "__main__":

    start_time = time.time()
    patient_number = 5
    clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Patient_models/subject_targets.csv"
    json_path = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Patient_models/pacs-scd-005/ROMSimulations/bc_optimization/bc_optimized.json"
    sv_project_filepath = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Patient_models/pacs-scd-{patient_number:03d}"
    particles_npy = "./SMC_results/smc_result_pacs005.npy"
    scores_npy = "./SMC_results/smc_scores_pacs005.npy"
    mapping_dict = {
        "L_ICA":0, "R_ICA":1, "L_VA":2, "R_VA":3, "L_SCA":4, "L_PCA":5,
        "L_MCA":6, "L_ACA":7, "R_ACA":8, "R_MCA":9, "R_PCA":10, "R_SCA":11,
    }
    num_particles = 2000
    num_cores = 16
    err_tolerance = 0.05

    clinical_data  = get_clinical_data(file=clinical_data_file, p_number=patient_number)
    #center_particles, limit_particles = sample_particles(particles_npy, scores_npy, n_center=2, n_limit=6)
    with open(json_path, 'r') as f: json_dict = json.load(f)
    active_rcr_ids = []
    deterministic_param_values = []
    for i in range(4, 12):
        bc_list = json_dict["boundary_conditions"]
        try:
            params = next(bc["bc_values"] for bc in bc_list if bc.get("bc_name") == f"RCR_{i}")
            for n, val in enumerate(params.values()):
                if n < 3:
                  deterministic_param_values.append(round(val,6))
            active_rcr_ids.append(i)
        except StopIteration:
            continue

        # Llamada directa
    # Llamada al script
    validate_confidence_region("SMC_results/smc_result_pacs005.npy", 
                           "SMC_results/smc_scores_pacs005.npy", 
                           json_path, active_rcr_ids, clinical_data, mapping_dict, n_samples=5)
    

