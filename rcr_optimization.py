import pysvzerod
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import copy 
from zeroDSims import plot_custom_0d_results

def return_rcr(json_dict, vessel_name):
    bc_list = json_dict["boundary_conditions"]
    for bc in bc_list:
        if bc.get("bc_name") == vessel_name:
            return bc["bc_values"]

def update_rcr(json_dict, new_Rp, new_C, new_Rd):
    bc_list = json_dict["boundary_conditions"]
    for bc in bc_list:
        if bc.get("bc_name") == "RCR_7":
            bc["bc_values"]["Rp"] = new_Rp
            bc["bc_values"]["Rd"]  = new_Rd
            bc["bc_values"]["C"] = new_C

class ThresholdReached(Exception):
    """Excepción lanzada cuando la optimización es lo suficientemente buena."""
    def __init__(self, params, error, p_mean, pulse):
        self.params = params
        self.error = error
        self.p_mean = p_mean
        self.pulse = pulse
        super().__init__()       


def objective_function(scaling_factors, base_params, json_dict, target_p, target_pulse):
    f_Rp, f_Rd, f_C = scaling_factors

    Rp = base_params['Rp'] * f_Rp
    Rd = base_params['Rd'] * f_Rd
    C  = base_params['C']  * f_C
    
    if Rd < Rp:
        return 1e6 + (Rp - Rd)
    
    try:
        json_to_run = copy.deepcopy(json_dict)
        update_rcr(json_dict=json_to_run, new_Rp=Rp, new_C=C, new_Rd=Rd)
        
        solver = pysvzerod.Solver(json_to_run)
        solver.run()
        
        df = solver.get_full_result()  
        branch_data = df[df['name'] == "branch7"]
        
        # Filtramos los últimos 200 puntos para asegurar estado estacionario
        pressure_mmhg = branch_data["pressure_out"].values / 133.3
        
        p_mean = np.mean(pressure_mmhg)
        pulse = np.max(pressure_mmhg) - np.min(pressure_mmhg)

        w_pmean = 2.0 
        w_pulse = 5.0 

        error_p = ((p_mean - target_p) / target_p)**2
        error_pulse = ((pulse - target_pulse) / target_pulse)**2
        total_error = (w_pmean * error_p) + (w_pulse * error_pulse)

        print(f"  Rp={Rp:.1f} Rd={Rd:.1f} C={C:.6f} | P_mean={p_mean:.1f} Pulse={pulse:.1f} | Error={total_error:.4f}")
        
        if total_error < 0.005:
            raise ThresholdReached(scaling_factors, total_error, p_mean, pulse)
            
        return total_error

    except ThresholdReached as e:
        raise e
    except Exception:
        return 1e10

def rcr_optimization(json_dict, target_p, target_pulse, n_restarts = 3):
    best_error = float('inf')
    base_params = copy.deepcopy(return_rcr(json_dict, "RCR_7"))
    bounds = [(0.1, 10.0), (0.1, 10.0), (0.0001, 2.0)]
    
    found_by_threshold = False
    final_scaling = [1.0, 1.0, 1.0]

    for i in range(n_restarts):
        if i == 0:
            initial_guess = [1.0, 1.0, 1.0]
        else:
            initial_guess = [random.uniform(0.5, 2.0), random.uniform(1.0, 5.0), random.uniform(0.001, 0.1)]
        
        print(f"\n Multistart iteration {i+1}/{n_restarts} ---")
        try:
            res = minimize(
                objective_function,
                initial_guess, 
                args=(base_params, json_dict, target_p, target_pulse),
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-4, 'maxiter': 30}
            )
            if res.fun < best_error:
                best_error = res.fun
                final_scaling = res.x

        except ThresholdReached as e:
            print(f"\n Threshold reached in iteration {i+1}!")
            print(f">>> Final error: {e.error:.6f} (< 0.005)")
            print(f">>> P_mean: {e.p_mean:.2f} mmHg | Pulse: {e.pulse:.2f} mmHg")
            final_scaling = e.params
            found_by_threshold = True
            break

    if not found_by_threshold:
        print(f"\n Number of iterations exceeded. Best error: {best_error:.6f}")

    f_Rp, f_Rd, f_C = final_scaling
    final_Rp = base_params['Rp'] * f_Rp
    final_Rd = base_params['Rd'] * f_Rd
    final_C  = base_params['C']  * f_C
    
    print("-" * 50)
    print(f"FINAL RESULTS:")
    print(f"Rp: {final_Rp:.2f} | Rd: {final_Rd:.2f} | C: {final_C:.8f}")
    print("-" * 50)
    
    return final_Rp, final_Rd, final_C
    

if __name__ == "__main__":
    json_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-005/Models/zeroD_simulation/zeroD_script.json"
    with open(json_file, 'r') as file:
        json_dict = json.load(file)

    final_Rp, final_Rd, final_C = rcr_optimization(json_dict, target_p=80, target_pulse=60)
    json_to_run = copy.deepcopy(json_dict)
    update_rcr(json_dict=json_to_run, new_Rp=final_Rp, new_C=final_C, new_Rd=final_Rd)
    
    solver = pysvzerod.Solver(json_to_run)
    solver.run()
    
    df = solver.get_full_result()  
    plot_custom_0d_results(df, ["branch7"], "pressure_out")