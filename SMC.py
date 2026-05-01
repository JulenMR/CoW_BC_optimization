import numpy as np
import particles
from particles import distributions as dists
from particles import state_space_models as ssm 
import json
import pysvzerod
import copy
from generate_json import get_clinical_data


class RCR_UQ(ssm.StateSpaceModel):
    def __init__(self, json_dict, active_rcr_ids, clinical_targets, branch_map, mapping_dict, **kwargs):
        super().__init__(**kwargs)
        self.json_base = json_dict
        self.active_ids = active_rcr_ids
        self.targets = clinical_targets # Contiene 'mean_p' y 'flows'
        self.branch_map = branch_map
        self.mapping_dict = mapping_dict
        self.num_outlets = len(active_rcr_ids)

    def PX0(self):
        # ORDEN: Rp, C, Rd
        priors = []
        for _ in range(self.num_outlets):
            priors.extend([
                dists.Uniform(0.1, 10.0),   # Rp
                dists.Uniform(0.001, 0.5),  # C
                dists.Uniform(0.5, 50.0)    # Rd
            ])
        return dists.Indep(*priors)

    def PX(self, t, xp):
        # Parámetros constantes con jitter mínimo para exploración
        return dists.Normal(loc=xp, scale=1e-4)

    def PY(self, t, xp, x):
        js = copy.deepcopy(self.json_base)
        # Pasamos x que ahora sigue el orden Rp, C, Rd
        update_all_outlets(js, x, self.active_ids)
        
        try:
            solver = pysvzerod.Solver(js)
            solver.run()
            df = solver.get_full_result()
        except Exception:
            return -np.inf # Penalización si la simulación falla

        sim_mean_p, sim_flows = get_stats(df, self.active_ids)

        y_sim = []
        y_target = []
        scales = []
        
        # Comparación de flujos (8 outlets)
        for b_name, csv_name in self.branch_map.items():
            branch_key = f"branch{self.mapping_dict[csv_name]}"
            if branch_key in sim_flows:
                y_sim.append(sim_flows[branch_key])
                y_target.append(self.targets['flows'][csv_name])
                # Tolerancia: 10% del valor clínico o mínimo 0.1 ml/s
                scales.append(max(0.1, 0.1 * self.targets['flows'][csv_name]))

        # Comparación de presión (Entrada)
        y_sim.append(sim_mean_p)
        y_target.append(self.targets['mean_p'])
        scales.append(2.0) # Tolerancia de 2 mmHg

        log_lik = dists.Normal(loc=np.array(y_sim), scale=np.array(scales)).logpdf(np.array(y_target))
        return np.sum(log_lik)

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
    p_means = []
    
    available_branches = df['name'].unique()
    for name in inlet_names:
        if name in available_branches:
            data = df[df['name'] == name]
            p_vals = data["pressure_in"].values / 133.32
            p_means.append(np.mean(p_vals))
    
    outlet_stats = {}
    for rcr_id in active_rcr_ids:
        name = f"branch{rcr_id}"
        if name in available_branches:
            data = df[df['name'] == name]
            outlet_stats[name] = np.mean(data["flow_out"].values)
        
    return np.mean(p_means), outlet_stats

if __name__ == "__main__":
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
               "flows": clinical_flows}

    rcr_uq = RCR_UQ(active_rcr_ids=active_rcr_ids, clinical_targets=targets, branch_map= mapping_dict)