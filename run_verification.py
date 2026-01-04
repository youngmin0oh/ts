import subprocess
import os
import re
import csv
import sys
import time

def run_comparison_experiments():
    # Configuration
    root_dir = os.getcwd()
    
    # Define script paths
    script_original = "Adapter-X+Y/run_xy_combined.py"
    script_norm = "Adapter-X+Y/run_xy_combined_norm.py"
    
    # Datasets and Models - FOCUSED for Verification
    datasets = ["ETTh2", "ETTm1"]
    models = ["iTransformer"]
    
    # Parameters
    delta = 0.01
    seq_len = 96
    pred_len = 96
    train_epochs = 1
    
    # Output File
    csv_file = os.path.join(root_dir, "comparison_experiment_results_verification.csv")
    
    print("Starting Focused Verification: Adaptive Norm")
    print("====================================================================")
    
    for dataset in datasets:
        for model in models:
            print(f"----------------------------------------------------------------")
            print(f"Dataset={dataset}, Model={model}, Delta={delta}")
            print(f"----------------------------------------------------------------")
            
            # Dictionary to store metrics
            metrics = {
                "baseline": {"mse": None, "mae": None},
                "add": {"mse": None, "mae": None},
                "mul": {"mse": None, "mae": None},
                "affine": {"mse": None, "mae": None},         # Original
                "affine_norm": {"mse": None, "mae": None}      # NEW: Adaptive Norm
            }
            
            # ------------------------------------------------------------------
            # Phase 1: Run Original Script (Add, Mul, Affine)
            # ------------------------------------------------------------------
            print(f"  > Phase 1: Running Original Adapters (Add, Mul, Affine)...")
            cmd_orig = [
                "python", "-u", script_original,
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
                "--train_epochs", str(train_epochs),
                "--batch_size", "32",
                "--adapter_mode", "affine", # Run only affine for speed
                "--checkpoints", "Adapter-X+Y/AdaCali/checkpoints/",
                "--gpu", "0"
            ]
            
            try:
                process = subprocess.Popen(cmd_orig, cwd=root_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                
                current_mode = None
                for line in process.stdout:
                    # Parse Baseline
                    if "mse:" in line and "mae:" in line and "total_loss" not in line and metrics["baseline"]["mse"] is None:
                         match = re.search(r"mse:([0-9.]+),\s*mae:([0-9.]+)", line)
                         if match:
                             metrics["baseline"]["mse"] = float(match.group(1))
                             metrics["baseline"]["mae"] = float(match.group(2))
                    
                    # Detect Mode
                    if "Running Adapter Mode:" in line:
                        if "add" in line: current_mode = "add"
                        elif "mul" in line: current_mode = "mul"
                        elif "affine" in line: current_mode = "affine"
                        
                    # Parse Adapter Result
                    if "final test_loss=" in line and "mae_loss=" in line:
                         match = re.search(r"final test_loss=([0-9.]+),\s*mae_loss=([0-9.]+)", line)
                         if match and current_mode:
                             metrics[current_mode]["mse"] = float(match.group(1))
                             metrics[current_mode]["mae"] = float(match.group(2))
                             
                process.wait()
            except Exception as e:
                print(f"    Error in Phase 1: {e}")

            # ------------------------------------------------------------------
            # Phase 2: Run Normalized Script (Affine-Norm) -> Actually Adaptive Norm now
            # ------------------------------------------------------------------
            print(f"  > Phase 2: Running Adaptive Norm (Affine-Norm)...")
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
                "--train_epochs", str(train_epochs),
                "--batch_size", "32",
                "--adapter_mode", "affine", # Treats 'affine' as 'affine_norm' in this script
                "--checkpoints", "Adapter-X+Y/AdaCali/checkpoints/",
                "--gpu", "0"
            ]
            
            try:
                process = subprocess.Popen(cmd_norm, cwd=root_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                
                for line in process.stdout:
                    # Verify Baseline Match (Optional, but good sanity check)
                    
                    # Parse Adapter Result
                    if "final test_loss=" in line and "mae_loss=" in line:
                         match = re.search(r"final test_loss=([0-9.]+),\s*mae_loss=([0-9.]+)", line)
                         if match:
                             metrics["affine_norm"]["mse"] = float(match.group(1))
                             metrics["affine_norm"]["mae"] = float(match.group(2))
                             
                process.wait()
            except Exception as e:
                print(f"    Error in Phase 2: {e}")

            # ------------------------------------------------------------------
            # Calculate Improvements & logging
            # ------------------------------------------------------------------
            row = {
                "Dataset": dataset, "Model": model, "Delta": delta,
                "Baseline_MSE": metrics["baseline"]["mse"],
                "Baseline_MAE": metrics["baseline"]["mae"]
            }
            
            def calc_imp(base, curr):
                if base is None or curr is None: return "N/A"
                return f"{(base - curr)/base * 100:.2f}%"

            # Add, Mul, Affine, Affine-Norm
            for m in ["add", "mul", "affine", "affine_norm"]:
                key_mse = f"{m.replace('_', '-').title()}_MSE" # e.g. Add_MSE, Affine-Norm_MSE
                key_mae = f"{m.replace('_', '-').title()}_MAE"
                key_imp = f"Imp_{m.replace('_', '-').title()}(%)"
                
                row[key_mse] = metrics[m]["mse"] if metrics[m]["mse"] else "N/A"
                row[key_mae] = metrics[m]["mae"] if metrics[m]["mae"] else "N/A"
                row[key_imp] = calc_imp(metrics["baseline"]["mse"], metrics[m]["mse"])
            
            # Save to CSV
            file_exists = os.path.isfile(csv_file)
            headers = ["Dataset", "Model", "Delta", "Baseline_MSE", "Baseline_MAE",
                       "Add_MSE", "Add_MAE", "Imp_Add(%)",
                       "Mul_MSE", "Mul_MAE", "Imp_Mul(%)",
                       "Affine_MSE", "Affine_MAE", "Imp_Affine(%)",
                       "Affine-Norm_MSE", "Affine-Norm_MAE", "Imp_Affine-Norm(%)"]
            
            try:
                with open(csv_file, mode='a' if file_exists else 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    if not file_exists: writer.writeheader()
                    writer.writerow(row)
                print(f"  > Saved results to {csv_file}")
            except Exception as e:
                print(f"  > Error saving CSV: {e}")

    print("\n=====================================================")
    print("Verification Experiment Finished.")

if __name__ == "__main__":
    run_comparison_experiments()
