import pysvzerod
import numpy as np
import json
import copy
from scipy.optimize import minimize
import pandas as pd
from generate_json import get_pressure

def get_clinical_flows(file, p_number):
    df = pd.read_csv(file)
    row = df[df['subject'] == p_number]
    if row.empty: raise ValueError(f"Patient {p_number} not found")
    
    data = row.iloc[0, 4:].to_dict()
    return {k.strip(): v for k, v in data.items()}


def return_rcr(json_dict, bc_name):
    bc_list = json_dict["boundary_conditions"]
    for bc in bc_list:
        if bc.get("bc_name") == bc_name:
            return bc["bc_values"]
    return None

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

def get_inlet_stats(df, inlet_names):
    stats = {}
    for name in inlet_names:
        data = df[df['name'] == name]
        p_values = data["pressure_in"].values / 133.322
        stats[name] = {
            "mean": np.mean(p_values),
            "pulse": np.max(p_values) - np.min(p_values)
        }
    return stats

def get_outlet_stats(df, outlet_names):
    stats = {}
    for name in outlet_names:
        data = df[df['name'] == name]
        stats[name] = np.mean(data["flow_out"].values)
    return stats


def monitor_callback(xk, base_params_list, json_dict):
    global iteration_count
    iteration_count += 1
    
    json_to_run = copy.deepcopy(json_dict)
    update_all_outlets(json_to_run, xk, base_params_list)
    
    solver = pysvzerod.Solver(json_to_run)
    solver.run()
    df = solver.get_full_result()
    
    inlet_names = ["branch0", "branch1", "branch2", "branch3"]
    stats = get_inlet_stats(df, inlet_names)
    
    avg_m = np.mean([stats[name]["mean"] for name in inlet_names])
    avg_p = np.mean([stats[name]["pulse"] for name in inlet_names])
    
    print(f"Iteration {iteration_count:02d} -> P_mean_avg: {avg_m:5.2f} | P_pulse_avg: {avg_p:5.2f}")

def monitor_callback_phase2(xk, base_params_list, json_dict, target_flows):
    global iteration_count
    iteration_count += 1
    
    # 1. Ejecutar simulación
    json_to_run = copy.deepcopy(json_dict)
    update_all_outlets(json_to_run, xk, base_params_list)
    
    solver = pysvzerod.Solver(json_to_run)
    solver.run()
    df = solver.get_full_result()
    
    # 2. Monitorizar Presión (Inlets)
    inlet_names = ["branch0", "branch1", "branch2", "branch3"]
    p_stats = get_inlet_stats(df, inlet_names)
    avg_m = np.mean([p_stats[name]["mean"] for name in inlet_names])
    
    # 3. Extraer flujos de los outlets
    outlet_names = list(BRANCH_MAPPING.keys())
    simulated_flows = get_outlet_stats(df, outlet_names)
    
    print(f"\n--- Iteration {iteration_count:02d} | Avg Pressure: {avg_m:5.2f} mmHg ---")
    print(f"{'Branch':<10} | {'Target (mm3/s)':<15} | {'Sim (mm3/s)':<12} | {'Error (%)':<8}")
    print("-" * 55)

    flow_errors = []
    for b_name, csv_name in BRANCH_MAPPING.items():
        # Target del CSV (convertido de mL/s a mm3/s)
        q_target = target_flows[csv_name] * 1000.0
        # Valor actual de la simulación
        q_sim = simulated_flows[b_name]
        
        # Calcular error individual
        err = abs(q_sim - q_target) / q_target * 100
        flow_errors.append(err)
        
        print(f"{csv_name:<10} | {q_target:>14.2f} | {q_sim:>11.2f} | {err:>7.1f}%")

    avg_flow_error = np.mean(flow_errors)
    print("-" * 55)
    print(f"TOTAL AVG FLOW ERROR: {avg_flow_error:6.2f}%")

def objective_function_1(scaling_factors, base_params_list, json_dict, target_p, target_pulse):
    try:
        json_to_run = copy.deepcopy(json_dict)
        update_all_outlets(json_to_run, scaling_factors, base_params_list)
        
        solver = pysvzerod.Solver(json_to_run)
        solver.run()
        df = solver.get_full_result()
        
        inlet_names = ["branch0", "branch1", "branch2", "branch3"]
        stats = get_inlet_stats(df, inlet_names)
        
        total_error = 0
        means = []
        pulses = []

        for name in inlet_names:
            m, p = stats[name]["mean"], stats[name]["pulse"]
            means.append(m)
            pulses.append(p)
            
            error_p = ((m - target_p) / target_p)**2
            error_pulse = ((p - target_pulse) / target_pulse)**2
            total_error += (1.0 * error_p) + (1.0 * error_pulse)

        return total_error
    except Exception:
        return 1e10
    
def objective_function_2(scaling_factors, base_params_list, json_dict, target_flows):
    try:
        json_to_run = copy.deepcopy(json_dict)
        update_all_outlets(json_to_run, scaling_factors, base_params_list)
        
        solver = pysvzerod.Solver(json_to_run)
        solver.run()
        df = solver.get_full_result()
        
        outlet_names = list(BRANCH_MAPPING.keys())
        simulated_flows = get_outlet_stats(df, outlet_names)
        
        total_error = 0
        for b_name, csv_name in BRANCH_MAPPING.items():
            q_target = target_flows[csv_name] * 1000.0
            q_sim = simulated_flows[b_name]
            
            total_error += ((q_sim - q_target) / q_target)**2

        penalty = 0
        for i in range(8):
            idx = i * 3
            f_Rp, f_Rd = scaling_factors[idx], scaling_factors[idx+1]
            if (f_Rp * base_params_list[i]['Rp']) > (0.2 * f_Rd * base_params_list[i]['Rd']):
                penalty += 100.0 

        return total_error + penalty
    except Exception as e:
        return 1e10

# --- PROCESO DE OPTIMIZACIÓN ---

def run_global_optimization(json_file_path, target_p, target_pulse, target_flows):
    with open(json_file_path, 'r') as file:
        json_dict = json.load(file)

    base_params_list = []
    for i in range(4, 12):
        params = return_rcr(json_dict, f"RCR_{i}")
        base_params_list.append(copy.deepcopy(params))

    bounds = [(0.7, 3.0), (0.7, 3.0), (0.5, 5.0)]*8
    bounds_2 = [(0.8, 1.3), (0.7, 3.0), (0.9, 1.1)]*8
    initial_guess = [1.0] * 24
    
    # --- PHASE 1: PRESSURE ---
    print(f"\n>>> PHASE 1: Inlet pressure setting | Mean pressure: {target_p}, Pulse: {target_pulse} ")
    res_phase1 = minimize(
        objective_function_1,
        initial_guess,
        args=(base_params_list, json_dict, target_p, target_pulse),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-3, 'maxiter': 30},
        callback=lambda xk: monitor_callback(xk, base_params_list, json_dict)
    )
    output_json_phase_1 = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/zeroD_script_phase1.json"
    phase1_json = copy.deepcopy(json_dict)
    update_all_outlets(phase1_json, res_phase1.x, base_params_list)
    with open(output_json_phase_1, "w") as f:
        json.dump(phase1_json, f, indent=4)

    # --- PHASE 2: FLOW ---
    print("\n>>> PHASE 2: Outlet flow setting")
    initial_conditions_phase2 = res_phase1.x
    res_phase2 = minimize(
        objective_function_2,
        initial_conditions_phase2, 
        args=(base_params_list, json_dict, target_flows),
        method='L-BFGS-B',
        bounds=bounds_2,
        options={'ftol': 1e-4, 'maxiter': 30},
        callback=lambda xk: monitor_callback_phase2(xk, base_params_list, json_dict, target_flows)
    )

    final_json = copy.deepcopy(json_dict)
    update_all_outlets(final_json, res_phase2.x, base_params_list)
    return final_json


if __name__ == "__main__":
    path_to_json = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/zeroD_script.json"
    clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/corrected_subject_targets.csv"
    output_json = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/zeroD_script_optimized.json"
    mean_p, pulse = get_pressure(p_file=clinical_data_file, p_number=8)
    clinical_flows = get_clinical_flows(file=clinical_data_file, p_number=8)
    iteration_count = 0

    BRANCH_MAPPING = {
    "branch4": "SCA_L", "branch5": "PCA_L", "branch6": "MCA_L", "branch7": "ACA_L",
    "branch8": "ACA_R", "branch9": "MCA_R", "branch10": "PCA_R", "branch11": "SCA_R"
    }
    
    optimized_json = run_global_optimization(path_to_json, target_p=mean_p, target_pulse=pulse, target_flows=clinical_flows)
    
    with open(output_json, "w") as f:
        json.dump(optimized_json, f, indent=4)
   
   
