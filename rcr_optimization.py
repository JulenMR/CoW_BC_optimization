import pysvzerod
import numpy as np
import json
import copy
from scipy.optimize import minimize

class OptimizatorState:
    def __init__(self):
        self.iteration = 0
        self.last_p_mean = 0.0
        self.last_p_pulse = 0.0
        self.last_flow_errors = {}
        self.last_total_flow_err = 0.0
        self.last_flows_sim = {}    
        self.last_flows_target = {} 

    def update_p(self, p_mean, p_pulse):
        self.last_p_mean = p_mean
        self.last_p_pulse = p_pulse

    def update_f(self, p_mean, p_pulse, flow_err_dict, total_err, sim_vals, target_vals):
        self.last_p_mean = p_mean
        self.last_p_pulse = p_pulse
        self.last_flow_errors = flow_err_dict
        self.last_total_flow_err = total_err
        self.last_flows_sim = sim_vals
        self.last_flows_target = target_vals

state = OptimizatorState()

def update_all_outlets(json_dict, scaling_factors, base_params_list, active_rcr_ids):
    bc_list = json_dict["boundary_conditions"]
    for i, rcr_id in enumerate(active_rcr_ids):
        idx = i * 3
        f_Rp, f_Rd, f_C = scaling_factors[idx : idx + 3]
        target_bc_name = f"RCR_{rcr_id}"
        
        for bc in bc_list:
            if bc.get("bc_name") == target_bc_name:
                bc["bc_values"]["Rp"] = base_params_list[i]['Rp'] * f_Rp
                bc["bc_values"]["Rd"] = base_params_list[i]['Rd'] * f_Rd
                bc["bc_values"]["C"]  = base_params_list[i]['C']  * f_C

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

# OBJECTIVE FUNCTIONS  
def objective_phase1(xk, base_params, json_dict, target_p, target_pulse, active_rcr_ids):
    try:
        js = copy.deepcopy(json_dict)
        update_all_outlets(js, xk, base_params, active_rcr_ids)
        solver = pysvzerod.Solver(js); solver.run()
        df = solver.get_full_result()
        
        m, p, _ = get_stats(df, active_rcr_ids)
        state.update_p(m, p)
        
        err = ((m - target_p)/target_p)**2 + ((p - target_pulse)/target_pulse)**2
        return err
    except: return 1e10

def objective_phase2(xk, base_params, json_dict, target_flows, target_p, target_pulse, BRANCH_MAP, active_rcr_ids):
    try:
        js = copy.deepcopy(json_dict)
        update_all_outlets(js, xk, base_params, active_rcr_ids)
        solver = pysvzerod.Solver(js); solver.run()
        df = solver.get_full_result()
        
        p_mean, p_pulse, sim_flows = get_stats(df, active_rcr_ids)
        
        flow_err_sum = 0
        individual_errors = {}
        sim_vals_log = {}
        target_vals_log = {}

        for b_name, csv_name in BRANCH_MAP.items():
            if b_name not in sim_flows: continue
            q_target = target_flows[csv_name] 
            q_sim = sim_flows[b_name]
            
            # Mean squared error
            err_sq = ((q_sim - q_target) / q_target)**2
            flow_err_sum += err_sq
            
            # Save the error percentage to visualize
            individual_errors[csv_name] = np.sqrt(err_sq) * 100
            sim_vals_log[csv_name] = q_sim
            target_vals_log[csv_name] = q_target

        p_penalty = 1.0 * ((p_mean - target_p) / target_p)**2
        pulse_penalty = 1.0 * ((p_pulse - target_pulse) / target_pulse)**2
        
        # Update state
        state.update_f(p_mean, p_pulse, individual_errors, 
                       np.mean(list(individual_errors.values())),
                       sim_vals_log, target_vals_log)
        
        return flow_err_sum + p_penalty + pulse_penalty
    except Exception as e: 
        print(f"Simulation error: {e}")
        return 1e10

# CALLBACKS
def cb_p1(xk):
    state.iteration += 1
    print(f"P1 | Iteration {state.iteration:02d} -> Mean pressure {state.last_p_mean:5.2f} | Pulse: {state.last_p_pulse:5.2f}")

def cb_p2(xk):
    state.iteration += 1
    print(f"\n{'='*55}")
    print(f"P2 | Iteration {state.iteration:02} | Mean_error: {state.last_total_flow_err:5.2f}%")
    print(f" Mean pressure: {state.last_p_mean:5.1f} | Pulse: {state.last_p_pulse:5.1f}")
    print(f"{'-'*55}")
    print(f"{'Vessel':<12} | {'Simulated':>12} | {'Target':>12} | {'Error %':>8}")
    print(f"{'-'*55}")
    
    for name in sorted(state.last_flows_sim.keys()):
        sim = state.last_flows_sim[name]
        target = state.last_flows_target[name]
        err = state.last_flow_errors[name]
        print(f"{name:<12} | {sim:12.2f} | {target:12.2f} | {err:7.1f}%")
    print(f"{'='*55}")

# Main function
def run_optimization(json_path, target_p, target_pulse, clinical_flows, BRANCH_MAP):
    with open(json_path, 'r') as f: json_dict = json.load(f)
    
    base_params = []
    active_rcr_ids = []
    for i in range(4, 12):
        bc_list = json_dict["boundary_conditions"]
        branch_count = 0
        try:
            params = next(bc["bc_values"] for bc in bc_list if bc.get("bc_name") == f"RCR_{i}")
            base_params.append(copy.deepcopy(params))
            active_rcr_ids.append(i)
        except StopIteration:
            continue

    branch_count = len(active_rcr_ids)
    print(f"Active outlets found: {branch_count} ({active_rcr_ids})")

    # Phase 1: Pressure
    initial_guess = [1.0] * (3*branch_count)
    bounds1 = [(0.7, 3.0), (0.7, 5.0), (0.2, 5.0)] * branch_count
    print(f"\n STARTING PHASE 1: Target mean pressure: {target_p:.2f} | Target pulse: {target_pulse:.2f}")
    state.iteration = 0
    res1 = minimize(objective_phase1, initial_guess, args=(base_params, json_dict, target_p, target_pulse, active_rcr_ids),
                    method='L-BFGS-B', bounds=bounds1, callback=cb_p1, options={'ftol': 1e-3})
    
    # Phase 2: Flow split + Pressure maintenance
    inlets = [0, 1, 2, 3]
    OUTLET_BR_MAP = {f"branch{num}": label for label, num in BRANCH_MAP.items() if num not in inlets}

    bounds2 = [(0.8, 1.2), (0.5, 5.0), (0.1, 10.0)] * branch_count
    print("\n STARTING PHASE 2: Flow split + maintaining pressure values")
    state.iteration = 0
    res2 = minimize(objective_phase2, res1.x, args=(base_params, json_dict, clinical_flows, target_p, target_pulse, OUTLET_BR_MAP, active_rcr_ids),
                    method='L-BFGS-B', bounds=bounds2, callback=cb_p2, options={'ftol': 1e-4})

    final_json = copy.deepcopy(json_dict)
    update_all_outlets(final_json, res2.x, base_params, active_rcr_ids)
    return final_json
