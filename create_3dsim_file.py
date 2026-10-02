import json
import re

def update_svfsi(json_path, inp_path, mapping_dict, viscosity_value=None, timestep_size=None, save_path=None):
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
            
            # INSERT "Add face:" block
            face_check_pattern = rf"Add face:\s*{re.escape(artery_name)}\b"
            if not re.search(face_check_pattern, inp_content):
                ref_face_pattern = r"(Add face:\s*R_SCA\s*\{[^}]*?Face file path:\s*mesh-complete/mesh-surfaces/cap_R_SCA\.vtp\s*\})"
                
                new_face_block = (
                    f"\n   Add face: {artery_name} {{\n"
                    f"      Face file path: mesh-complete/mesh-surfaces/{artery_name}.vtp\n"
                    f"   }}"
                )
                
                if re.search(ref_face_pattern, inp_content, re.DOTALL):
                    inp_content = re.sub(ref_face_pattern, r"\1" + new_face_block, inp_content, count=1, flags=re.DOTALL)
                    print(f"Added face block for {artery_name}")
                else:
                    print(f"Warning: Reference face block for R_SCA not found. Couldn't add face for {artery_name}.")

            # INSERT rcr BLOCK
            bc_pattern = rf"(Add BC:\s*{re.escape(artery_name)}\s*\{{[^}}]*?RCR values:\s*\()[^\)]+(\))"
            replacement = rf"\g<1>{rp:.6f}, {c:.8f}, {rd:.6f}\g<2>"
            
            if re.search(bc_pattern, inp_content, re.DOTALL):
                inp_content = re.sub(bc_pattern, replacement, inp_content, flags=re.DOTALL)
                print(f"Updated BC for {artery_name}")
            else:
                ref_bc_pattern = r"(Add BC:\s*R_SCA\s*\{[^}]*?\})"
                
                new_bc_block = (
                    f"\n\n   Add BC: {artery_name} {{\n"
                    f"      Type: Neumann\n"
                    f"      Time dependence: RCR\n"
                    f"      RCR values: ({rp:.6f}, {c:.8f}, {rd:.6f})\n"
                    f"      Distal pressure: 0.0\n"
                    f"   }}"
                )
                
                if re.search(ref_bc_pattern, inp_content, re.DOTALL):
                    # Insert right after the R_SCA block (keeping it inside the main closing braces)
                    inp_content = re.sub(ref_bc_pattern, r"\1" + new_bc_block, inp_content, count=1, flags=re.DOTALL)
                    print(f"Added new BC block inside section for {artery_name}")
                else:
                    print(f"Warning: Reference BC block for R_SCA not found. Couldn't insert BC for {artery_name}.")

    # Update Time step size
    timestep_size_pattern = r"(Time step size:\s*)([\d\.\-\+eE]+)"
    if timestep_size is not None:
        new_value = timestep_size / 1000.0
        timestep_size_replacement = rf"\g<1>{new_value:.6f}"
        
        if re.search(timestep_size_pattern, inp_content):
            inp_content = re.sub(timestep_size_pattern, timestep_size_replacement, inp_content)
            print(f"Updated timestep size to: {new_value:.6f}")
        else:
            print("Warning: 'Time step size' block not found in the .inp file.")

    # Update Viscosity
    viscosity_pattern = r"(Viscosity: Constant \{Value:\s*)([\d\.\-\+eE]+)"
    if viscosity_value is not None:
        val_float = float(viscosity_value)
        viscosity_replacement = rf"\g<1>{val_float:.6f}"
        
        if re.search(viscosity_pattern, inp_content):
            inp_content = re.sub(viscosity_pattern, viscosity_replacement, inp_content)
            print(f"Updated Viscosity to: {val_float:.6f}")
        else:
            print("Warning: 'Viscosity: Constant {Value: ...}' block not found in the .inp file.")

    save_path = save_path or inp_path
    with open(save_path, 'w') as f:
        f.write(inp_content)
    
    print(f"\nFile saved: {save_path}")