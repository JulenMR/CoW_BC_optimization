import pysvzerod
import numpy as np
import json
import copy
from scipy.optimize import minimize
import pandas as pd
from generate_json import get_pressure


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
        p_values = data["pressure_in"].values / 1333.322
        stats[name] = {
            "mean": np.mean(p_values),
            "pulse": np.max(p_values) - np.min(p_values)
        }
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

def objective_function(scaling_factors, base_params_list, json_dict, target_p, target_pulse):
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
            total_error += (2.0 * error_p) + (4.0 * error_pulse)

        return total_error
    except Exception:
        return 1e10

# --- PROCESO DE OPTIMIZACIÓN ---

def run_global_optimization(json_file_path, target_p=85, target_pulse=50):
    with open(json_file_path, 'r') as file:
        json_dict = json.load(file)

    base_params_list = []
    for i in range(4, 12):
        params = return_rcr(json_dict, f"RCR_{i}")
        base_params_list.append(copy.deepcopy(params))

    bounds = [(0.1, 10.0), (0.1, 10.0), (0.01, 10.0)] * 8
    initial_guess = [1.0] * 24

    print(f"{'='*60}")
    print(f" OPTIMIZING FOR TARGET: Mean={target_p} / Pulse={target_pulse}")
    print(f"{'='*60}")
    
    res = minimize(
        objective_function,
        initial_guess,
        args=(base_params_list, json_dict, target_p, target_pulse),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-4, 'maxiter': 40},
        callback=lambda xk: monitor_callback(xk, base_params_list, json_dict)
    )

    final_json = copy.deepcopy(json_dict)
    update_all_outlets(final_json, res.x, base_params_list)
    
    # Final results
    final_solver = pysvzerod.Solver(final_json)
    final_solver.run()
    final_stats = get_inlet_stats(final_solver.get_full_result(), ["branch0", "branch1", "branch2", "branch3"])

    print("\n" + "="*40)
    print("Final pressure in inlets")
    print("="*40)
    for name, s in final_stats.items():
        print(f"{name.upper():<10} | Media: {s['mean']:>6.2f} mmHg | Pulso: {s['pulse']:>6.2f} mmHg")
    print("="*40)
    
    return final_json



if __name__ == "__main__":
    path_to_json = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-005/Models/zeroD_simulation/zeroD_script.json"
    pressure_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/subject_targets.csv"
    mean_p, pulse = get_pressure(p_file=pressure_data_file, p_number=5)
    iteration_count = 0
   
    optimized_json = run_global_optimization(path_to_json, target_p=mean_p, target_pulse=pulse)
    
    with open("zeroD_optimized.json", "w") as f:
        json.dump(optimized_json, f, indent=4)
