import pysvzerod
import json
import pandas as pd
import importlib.util
import sys
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
from generate_json import get_clinical_data

cap_names = sorted([
        "L_SCA", "R_SCA", 
        "L_PCA", "R_PCA", 
        "L_MCA", "R_MCA", 
        "L_ACA", "R_ACA", 
        "L_ICA", "R_ICA", 
        "L_VA", "R_VA"
    ])

inlets = ['branch0', 'branch1', 'branch2', 'branch3'] 
outlets = ['branch4', 'branch5', 'branch6', 'branch7', 'branch8', 'branch9', 'branch10', 'branch11'] 
total = inlets + outlets
color_list = cm.get_cmap('tab10')(np.linspace(0, 1, len(cap_names)))
COLOR_MAP = dict(zip(cap_names, color_list))
BRANCH_MAPPING = {
    "branch0": "L_ICA", "branch1": "R_ICA", "branch2": "L_VA", "branch3": "R_VA", "branch4": "L_SCA", "branch5": "L_PCA", 
    "branch6": "L_MCA", "branch7": "L_ACA", "branch8": "R_ACA", "branch9": "R_MCA", "branch10": "R_PCA", "branch11": "R_SCA"
    }

order_idx = {name: i for i, name in enumerate(cap_names)}
total_ordered = sorted(total, key=lambda b: order_idx.get(BRANCH_MAPPING[b], 99))

def plot_combined_0d_3d_results(zeroD_json_filepath, filepath_3D, branchnames, parameter, mapping, title, save_path = None):
    
    # 1. Execute 0D simulation
    zeroD_json = json.load(open(zeroD_json_filepath))
    print("Executing 0D simulation...")
    solver = pysvzerod.Solver(zeroD_json)
    solver.run()
    df_zeroD = solver.get_full_result()
    
    # 2. Load 3D data
    flow_3d_opt_path = os.path.join(filepath_3D, "B_NS_Velocity_flux.txt")
    pressure_3d_opt_path = os.path.join(filepath_3D, "B_NS_Pressure_average.txt")
    
    # Select the 3D file based on the requested parameter
    file_3d = pressure_3d_opt_path if "pressure" in parameter else flow_3d_opt_path
    df_3d = pd.read_csv(file_3d, sep='\s+')
    # Extract the last 1000 points to analyze the stabilized cycle
    df_3d_last = df_3d.tail(1000).copy()
    
    # 3. Setup figure with 2 subplots sharing the Y-axis
    fig, (ax0, ax3) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    is_pressure = "pressure" in parameter
    unit_label = " (mmHg)" if is_pressure else " (mL/s)"

    found_any = False

    for name in branchnames:
        label_name = mapping[name]
        line_color = COLOR_MAP.get(label_name, "grey")
        
        # --- 0D PLOT (Left Subplot) ---
        branch_data_0d = df_zeroD[df_zeroD['name'] == name]
        if not branch_data_0d.empty:
            y_0d = branch_data_0d[parameter].values.copy()
            t_0d = branch_data_0d['time'].values
            
            # Unit conversion
            if is_pressure:
                y_0d /= 133.3 # Barye to mmHg
            else:
                y_0d /= 1000.0 # mm3/s to mL/s

            ax0.plot(t_0d, y_0d, label=label_name, linewidth=2, color=line_color)
            
            # --- 3D PLOT (Right Subplot) ---
            if label_name in df_3d_last.columns:
                y_3d = df_3d_last[label_name].values.copy()
                
                # Create a synthetic time axis for 3D results aligned with 0D cycle
                t_3d = np.linspace(t_0d.min(), t_0d.max(), len(y_3d))
                
                if not is_pressure:
                    # Use absolute value for flows (inlets are negative in SimVascular)
                    y_3d = np.abs(y_3d)/1000.0 
                
                ax3.plot(t_3d, y_3d, label=label_name, linewidth=2, color=line_color)
            
            found_any = True

    if not found_any:
        print("No branches found to plot.")
        plt.close()
        return

    # Left Subplot configuration (0D)
    ax0.set_title(f"0D Solver: {parameter}")
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel(parameter + unit_label)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(loc='upper right', fontsize='small', ncol=2)

    # Right Subplot configuration (3D)
    ax3.set_title(f"3D FSI: {parameter}")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Main figure title
    plt.suptitle(title, fontsize=16)
    # Adjust layout to prevent overlap with the main title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    if save_path:
        output_name = f"combined_0D_3D_{parameter}.png"
        plt.savefig(os.path.join(save_path, output_name), dpi=300)
    
    plt.show()


def simulation_comparison(zeroD_filepath, optimized_3d_filepath, manual_3d_filepath, clinical_data_filepath, save_path = None):

    # Run 0D solver
    zeroD_json = json.load(open(zeroD_filepath))
    print("Executing 0D simulation")
    solver = pysvzerod.Solver(zeroD_json)
    solver.run()
    df_zeroD = solver.get_full_result()

    # Load clinical data
    target_p, target_pulse, clinical_flows = get_clinical_data(file=clinical_data_filepath, p_number=11)

    # Load optimized 3D simulated data
    flow_3d_opt_path = os.path.join(optimized_3d_filepath, "B_NS_Velocity_flux.txt")
    opt_flow_3d = pd.read_csv(flow_3d_opt_path, sep='\s+')
    df_opt_flow_3d_last = opt_flow_3d.tail(1000)

    # Load manual 3D simulated data
    flow_3d_manual_path = os.path.join(manual_3d_filepath, "B_NS_Velocity_flux.txt")
    manual_flow_3d = pd.read_csv(flow_3d_manual_path, sep='\s+')
    df_manual_flow_3d_last = manual_flow_3d.tail(1000)

    print("\n>>> COMPARISON: Clinical data vs Optimized 0D vs Optimized 3D vs Manual 3D")
    print(f"{'Vessel':<12} | {'Clinical':>10} | {'0D Mean':>10} | {'Opt 3D':>10} | {'Manual 3D':>10} | {'Err manual%':>12} | {'Err optimized%':>12}")
    print("-" * 95)

    full_comparison = []

    for b_id, cap_name in BRANCH_MAPPING.items():
        if b_id in outlets: 
            # 0D Flow
            col_0d = "flow_in" if b_id in inlets else "flow_out"
            mean_0d = df_zeroD[df_zeroD['name'] == b_id][col_0d].mean() / 1000.0

            # 3D Flows (Converting to mL/s)
            mean_3d_opt = np.abs(df_opt_flow_3d_last[cap_name].mean() / 1000.0) if cap_name in df_opt_flow_3d_last.columns else np.nan
            mean_3d_manual = np.abs(df_manual_flow_3d_last[cap_name].mean() / 1000.0) if cap_name in df_manual_flow_3d_last.columns else np.nan
            
            # Clinical Flow
            q_clinical = clinical_flows.get(cap_name, 0.0) / 1000.0
            
            # Error calculations
            zeroD_err = abs(mean_0d - q_clinical) / q_clinical * 100 if q_clinical > 0 else 0
            opt_err = abs(mean_3d_opt - q_clinical) / q_clinical * 100 if q_clinical > 0 else 0
            manual_err = abs(mean_3d_manual - q_clinical) / q_clinical * 100 if q_clinical > 0 else 0

            print(f"{cap_name:<12} | {q_clinical:10.4f} | {mean_0d:10.4f} | {mean_3d_opt:10.4f} | {mean_3d_manual:10.4f} | {manual_err:11.1f}% | {opt_err:11.1f}% ")
            
            full_comparison.append({
                "Vessel": cap_name,
                "Clinical": q_clinical,
                "0D": mean_0d,
                "Opt_3D": mean_3d_opt,
                "Manual_3D": mean_3d_manual,
                "ZeroD_Err": zeroD_err,
                "Opt_Err": opt_err,
                "Manual_Err": manual_err
            })
        else:
            continue

    # Barplot
    df_plot = pd.DataFrame(full_comparison)
    plt.figure(figsize=(16, 7))

    x = np.arange(len(df_plot))
    width = 0.2  # Reduced width to fit 4 bars comfortably

    # Adjusting X positions: -1.5w, -0.5w, 0.5w, 1.5w to center the group
    plt.bar(x - 1.5*width, df_plot['Clinical'],  width, label='Clinical target', color='#2ecc71', alpha=0.8)
    plt.bar(x - 0.5*width, df_plot['0D'],        width, label='0D simulation',  color='#3498db', alpha=0.8)
    plt.bar(x + 0.5*width, df_plot['Opt_3D'],    width, label='Optimized 3D',   color='#e74c3c', alpha=0.8)
    plt.bar(x + 1.5*width, df_plot['Manual_3D'], width, label='Manual 3D',      color='#f1c40f', alpha=0.8)

    # Añadir etiquetas de error sobre las barras
    for i in range(len(df_plot)):
        # Text for 0D error
        if not np.isnan(df_plot['0D'][i]):
            plt.text(x[i] - 0.5*width, df_plot['0D'][i] + 0.01, 
                     f"{df_plot['ZeroD_Err'][i]:.1f}%", 
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='black', rotation=0)
            
        # Text for 3D optimization
        if not np.isnan(df_plot['Opt_3D'][i]):
            plt.text(x[i] + 0.5*width, df_plot['Opt_3D'][i] + 0.01, 
                     f"{df_plot['Opt_Err'][i]:.1f}%", 
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='black', rotation=0)
        
        # Text for manual 3D
        if not np.isnan(df_plot['Manual_3D'][i]):
            plt.text(x[i] + 1.5*width, df_plot['Manual_3D'][i] + 0.01, 
                     f"{df_plot['Manual_Err'][i]:.1f}%", 
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='black', rotation=0)

    plt.ylabel('Mean Flow (mL/s)', fontsize=12)
    plt.title('Validation Results: Clinical vs 0D vs Optimized 3D vs Manual 3D', fontsize=14)
    plt.xticks(x, df_plot['Vessel'], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    # plt.savefig(os.path.join(save_path, "total_comparison_validation.png"), dpi=300)
    plt.show()

if __name__ == "__main__":

    zeroD_json_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-011/Models/zeroD_simulation_2/zeroD_script_optimized.json"
    opt_3d_result = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-011/Simulations/fine/11_zeroD_opt/122-procs"
    manual_result_3d = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-011/Simulations/fine/11_asl/96-procs"
    clinical_data_file = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/subject_targets.csv"
    save_path = "/home/julenmr/Documents/CMU/Automatic_BC/Synthetic_data/Laras_models/pacs-scd-008/Models/zeroD_simulation/"

    # simulation_comparison(zeroD_filepath=zeroD_json_file, optimized_3d_filepath=opt_3d_result, manual_3d_filepath=manual_result_3d, 
    #                       clinical_data_filepath=clinical_data_file)
    plot_combined_0d_3d_results(zeroD_json_filepath = zeroD_json_file, filepath_3D = opt_3d_result, branchnames = inlets, 
                                parameter = "flow_in", mapping = BRANCH_MAPPING, title = "PACS011 inlet compatison")