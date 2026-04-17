import pysvzerod
import numpy as np
import json
import copy
from scipy.optimize import minimize
import pandas as pd
from generate_json import get_clinical_data

class OptimizatorState:
    def __init__(self):
        self.iteration = 0
        self.last_p_mean = 0.0
        self.last_p_pulse = 0.0
        self.last_flow_errors = {}
        self.last_total_flow_err = 0.0

    def update_p(self, m, p):
        self.last_p_mean = m
        self.last_p_pulse = p

    def update_f(self, p_mean, p_pulse, flow_dict, total_err):
        self.last_p_mean = p_mean
        self.last_p_pulse = p_pulse
        self.last_flow_errors = flow_dict
        self.last_total_flow_err = total_err

state = OptimizatorState()


def update_all_outlets(json_dict, scaling_factors, base_params_list):
    bc_list = json_dict["boundary_conditions"]
    for i in range(8): 
        idx = i * 3
        f_Rp, f_Rd, f_C = scaling_factors[idx : idx + 3]
        target_bc_name = f"RCR_{i+4}"
        for bc in bc_list:
            if bc.get("bc_name") == target_bc_name:
                bc["bc_values"]["Rp"] = base_params_list[i]['Rp'] * f_Rp
                bc["bc_values"]["Rd"] = base_params_list[i]['Rd'] * f_Rd
                bc["bc_values"]["C"]  = base_params_list[i]['C']  * f_C

def get_stats(df):
    inlet_names = ["branch0", "branch1", "branch2", "branch3"]
    p_means, p_pulses = [], []
    for name in inlet_names:
        data = df[df['name'] == name]
        p_vals = data["pressure_in"].values / 133.32 # mm-mg-s to mmHg
        p_means.append(np.mean(p_vals))
        p_pulses.append(np.max(p_vals) - np.min(p_vals))
    
    outlet_stats = {}
    for i in range(4, 12):
        name = f"branch{i}"
        data = df[df['name'] == name]
        outlet_stats[name] = np.mean(data["flow_out"].values)
        
    return np.mean(p_means), np.mean(p_pulses), outlet_stats

# OBJECTIVE FUNCTIONS  

def objective_phase1(xk, base_params, json_dict, target_p, target_pulse):
    try:
        js = copy.deepcopy(json_dict)
        update_all_outlets(js, xk, base_params)
        solver = pysvzerod.Solver(js); solver.run()
        df = solver.get_full_result()
        
        m, p, _ = get_stats(df)
        state.update_p(m, p)
        
        err = ((m - target_p)/target_p)**2 + ((p - target_pulse)/target_pulse)**2
        return err
    except: return 1e10

def objective_phase2(xk, base_params, json_dict, target_flows, target_p, target_pulse, BRANCH_MAP):
    try:
        js = copy.deepcopy(json_dict)
        update_all_outlets(js, xk, base_params)
        solver = pysvzerod.Solver(js); solver.run()
        df = solver.get_full_result()
        
        p_mean, p_pulse, sim_flows = get_stats(df)
        
        flow_err_sum = 0
        individual_errors = {}
        for b_name, csv_name in BRANCH_MAP.items():
            q_target = target_flows[csv_name] * 1000.0
            q_sim = sim_flows[b_name]
            err = ((q_sim - q_target) / q_target)**2
            flow_err_sum += err
            individual_errors[csv_name] = np.sqrt(err) * 100

        # Pressure and pulse penalization
        p_penalty = 1.0 * ((p_mean - target_p) / target_p)**2
        pulse_penalty = 1.0 * ((p_pulse - target_pulse) / target_pulse)**2
        
        state.update_f(p_mean, p_pulse, individual_errors, np.mean(list(individual_errors.values())))
        
        return flow_err_sum + p_penalty + pulse_penalty
    except: return 1e10

# CALLBACKS LIGEROS

def cb_p1(xk):
    state.iteration += 1
    print(f"P1 | Iter {state.iteration:02d} -> P_mean: {state.last_p_mean:5.2f} | P_pulse: {state.last_p_pulse:5.2f}")

def cb_p2(xk):
    state.iteration += 1
    print(f"P2 | Iter {state.iteration:02d} | P_avg: {state.last_p_mean:5.1f} | P_pulse: {state.last_p_pulse:5.1f} | Flow Err: {state.last_total_flow_err:5.2f}%")

# MAIN PIPELINE

def run_optimization(json_path, target_p, target_pulse, clinical_flows, BRANCH_MAP):
    with open(json_path, 'r') as f: json_dict = json.load(f)
    
    base_params = []
    for i in range(4, 12):
        bc_list = json_dict["boundary_conditions"]
        params = next(bc["bc_values"] for bc in bc_list if bc.get("bc_name") == f"RCR_{i}")
        base_params.append(copy.deepcopy(params))

    # Phase 1: Pressure
    initial_guess = [1.0] * 24
    bounds1 = [(0.7, 3.0), (0.7, 5.0), (0.2, 5.0)] * 8
    print(f"\n>>> STARTING PHASE 1: PRESSURE: MEAN: {target_p} | PULSE: {target_pulse}")
    state.iteration = 0
    res1 = minimize(objective_phase1, initial_guess, args=(base_params, json_dict, target_p, target_pulse),
                    method='L-BFGS-B', bounds=bounds1, callback=cb_p1, options={'ftol': 1e-3})
    


    # Phase 2: Flow split + Pressure maintenance
    bounds2 = [(0.8, 1.2), (0.5, 5.0), (0.1, 10.0)] * 8 
    print("\n>>> STARTING PHASE 2: FLOW SPLIT & PRESSURE MAINTENANCE")
    state.iteration = 0
    res2 = minimize(objective_phase2, res1.x, args=(base_params, json_dict, clinical_flows, target_p, target_pulse, BRANCH_MAP),
                    method='L-BFGS-B', bounds=bounds2, callback=cb_p2, options={'ftol': 1e-4})

    final_json = copy.deepcopy(json_dict)
    update_all_outlets(final_json, res2.x, base_params)
    return final_json

if __name__ == "__main__":
    patient_number = 11
    path_to_json = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Models/zeroD_simulation/zeroD_script.json"
    clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/corrected_subject_targets.csv"
    output_json = f"/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-{patient_number:03d}/Models/zeroD_simulation/zeroD_script_optimized.json"
    mean_p, pulse, clinical_flows = get_clinical_data(file=clinical_data_file, p_number=patient_number)
    iteration_count = 0

    BRANCH_MAPPING = {
    "branch4": "SCA_L", "branch5": "PCA_L", "branch6": "MCA_L", "branch7": "ACA_L",
    "branch8": "ACA_R", "branch9": "MCA_R", "branch10": "PCA_R", "branch11": "SCA_R"
    }


    opt_json = run_optimization(path_to_json, mean_p, pulse, clinical_flows, BRANCH_MAPPING)
    
    with open(output_json, "w") as f:
        json.dump(opt_json, f, indent=4)
    print(f"\nOptimization finalized. JSON File saved in {output_json}")