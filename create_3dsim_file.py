import json
import re
import os

def update_svfsi(json_path, inp_path, mapping_dict, save_path=None):
    with open(json_path, 'r') as f:
        zero_d_data = json.load(f)
    
    optimized_rcr = {}
    for bc in zero_d_data.get("boundary_conditions", []):
        if bc["bc_type"] == "RCR":
            vals = bc["bc_values"]
            optimized_rcr[bc["bc_name"]] = (vals["Rp"], vals["C"], vals["Rd"])
    
    with open(inp_path, 'r') as f:
        inp_content = f.read()

    for artery_name, branch_id in mapping_dict.items():
        rcr_key = f"RCR_{branch_id}"
        
        if rcr_key in optimized_rcr:
            rp, c, rd = optimized_rcr[rcr_key]
            
            pattern = rf"(Add BC:\s*{artery_name}\s*\{{[^}}]*?RCR values:\s*\()[^\)]+(\))"
            replacement = rf"\g<1>{rp:.6f}, {c:.8f}, {rd:.6f}\g<2>"
            
            if re.search(pattern, inp_content, re.DOTALL):
                inp_content = re.sub(pattern, replacement, inp_content, flags=re.DOTALL)
                print(f"Updated {artery_name}")
            else:
                print(f"Warning: BC block for {artery_name} not found.")

    with open(save_path, 'w') as f:
        f.write(inp_content)
    
    print(f"\n File saved: {save_path}")
