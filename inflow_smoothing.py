import numpy as np
import matplotlib.pyplot as plt
import os 
from scipy.interpolate import CubicSpline

def smooth_inflow(original_flow_file, optimization_file, threeDsim_file):     

    flow_files = {
        "LICA": os.path.join(original_flow_file, "LICA.dat"),
        "RICA": os.path.join(original_flow_file, "RICA.dat"),
        "LVA": os.path.join(original_flow_file, "LVA.dat"),
        "RVA": os.path.join(original_flow_file, "RVA.dat")
    }

    plt.figure(figsize=(12, 8))

    N_POINTS_OUT = 100  
    FOURIER_MODES = 30

    for i, (label, path) in enumerate(flow_files.items()):
        if os.path.exists(path):
            data = np.loadtxt(path, skiprows=1)
            timestep_size = data[-1,0]
            sort_idx = np.argsort(data[:, 0])
            time_pts = data[sort_idx, 0]
            flow_pts = np.abs(data[sort_idx, 1])

            t_last2, t_last = time_pts[-2], time_pts[-1]
            f_last2, f_last = flow_pts[-2], flow_pts[-1]
            synthetic_point_num = 2
            t_regression = np.linspace(t_last2, t_last, synthetic_point_num + 2)[1:-1]
            f_regression = f_last2 + (t_regression - t_last2) * (f_last - f_last2) / (t_last - t_last2)
            
            time_pts_augmented = np.concatenate([time_pts[:-1], t_regression, [t_last]])
            flow_pts_augmented = np.concatenate([flow_pts[:-1], f_regression, [f_last]])

            # Spline
            try:
                cs = CubicSpline(time_pts_augmented, flow_pts_augmented, bc_type='not-a-knot')
            except ValueError:
                cs = CubicSpline(time_pts_augmented, flow_pts_augmented, bc_type='clamped')
            
            time_smooth = np.linspace(time_pts.min(), time_pts.max(), N_POINTS_OUT)
            flow_smooth = cs(time_smooth)
            
            output_path_3d = os.path.join(threeDsim_file, f"{label}_3d_smooth.dat") 
            output_path_0d = os.path.join(optimization_file, f"{label}_0d_smooth.dat")
            
            with open(output_path_3d, 'w') as f:
                f.write(f"{N_POINTS_OUT},{FOURIER_MODES}\n")
                for t, fl in zip(time_smooth, flow_smooth):
                    f.write(f"{t:.6e}\t{-abs(fl):.6e}\n") 
            
            with open(output_path_0d, 'w') as f:
                for t, fl in zip(time_smooth, flow_smooth):
                    f.write(f"{t:.6e}\t{abs(fl):.6e}\n")



  