import subprocess
import os
import re

def run_verification():
    # Configuration
    root_dir = os.getcwd()
    script_norm = "Adapter-X+Y/run_xy_combined_norm.py"
    
    dataset = "ETTh2"
    model = "FreTS"
    delta = 0.01
    seq_len = 96
    pred_len = 96
    
    print(f"Starting Verification: {dataset} / {model} / Affine-Norm (Per-Channel)")
    
    cmd_norm = [
        "python", "-u", script_norm,
        "--is_training", "1",
        "--root_path", "./datasets/",
        "--data_path", f"{dataset}.csv",
        "--model_id", f"{dataset}_{seq_len}_{pred_len}",
        "--model", model,
        "--data", dataset,
        "--features", "M",
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--e_layers", "2",
        "--d_layers", "1",
        "--factor", "3",
        "--des", "Exp",
        "--itr", "1",
        "--delta", str(delta),
        "--learning_rate", "0.0001",
        "--train_epochs", "1",
        "--batch_size", "32",
        "--adapter_mode", "affine", 
        "--checkpoints", "Adapter-X+Y/AdaCali/checkpoints/",
        "--gpu", "0"
    ]
    
    try:
        process = subprocess.Popen(cmd_norm, cwd=root_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        
        baseline_mse = 0.192594 # From original CSV
        affine_norm_mse = None
        
        for line in process.stdout:
            print(line, end='') # Stream output to see progress
            
            if "final test_loss=" in line and "mae_loss=" in line:
                 match = re.search(r"final test_loss=([0-9.]+),\s*mae_loss=([0-9.]+)", line)
                 if match:
                     affine_norm_mse = float(match.group(1))
                     
        process.wait()
        
        print("\nVerification Results:")
        print(f"Baseline MSE (ETTh2/FreTS): {baseline_mse}")
        if affine_norm_mse:
            print(f"Affine-Norm MSE (New): {affine_norm_mse}")
            imp = (baseline_mse - affine_norm_mse) / baseline_mse * 100
            print(f"Improvement: {imp:.2f}%")
            if imp > -0.31: # Previous degraded result
                print("SUCCESS: Performance improved compared to previous degradation.")
            else:
                print("FAILURE: Performance is still degraded.")
        else:
            print("Could not parse Affine-Norm MSE.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_verification()
