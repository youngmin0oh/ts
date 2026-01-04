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
    
    # Datasets and Models - REDUCED SCOPE
    datasets = ["ETTh2"]
    models = ["iTransformer"]
    
    # Parameters
    delta = 0.01
    seq_len = 96
    pred_len = 96
    train_epochs = 1
    
    # Output File
    csv_file = os.path.join(root_dir, "test_norm_avg.csv")
    
    # Multi-Horizon Settings - REDUCED
    pred_lens = [96, 192]
    
    print("Starting Comparison Experiments: Add vs Mul vs Affine vs Affine-Norm (Multi-Horizon TEST)")
    print("====================================================================")
    
    for dataset in datasets:
        for model in models:
            
            # Storage for horizon results
            horizon_metrics = {
                "baseline": {"mse": [], "mae": []},
                "add": {"mse": [], "mae": []},
                "mul": {"mse": [], "mae": []},
                "affine": {"mse": [], "mae": []},
                "affine_norm": {"mse": [], "mae": []}
            }

            for pred_len in pred_lens:
                print(f"----------------------------------------------------------------")
                print(f"Dataset={dataset}, Model={model}, Delta={delta}, PredLen={pred_len}")
                print(f"----------------------------------------------------------------")
                
                # Dictionary to store metrics for current horizon
                metrics = {
                    "baseline": {"mse": None, "mae": None},
                    "add": {"mse": None, "mae": None},
                    "mul": {"mse": None, "mae": None},
                    "affine": {"mse": None, "mae": None},
                    "affine_norm": {"mse": None, "mae": None}
                }
                
                # ------------------------------------------------------------------
                # Phase 1: Run Original Script (Add, Mul, Affine)
                # ------------------------------------------------------------------
                # print(f"  > Phase 1: Running Original Adapters (Add, Mul, Affine)...")
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
                    "--adapter_mode", "all", # Run all standard modes
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
                # Phase 2: Run Normalized Script (Affine-Norm)
                # ------------------------------------------------------------------
                # print(f"  > Phase 2: Running Normalized Adapter (Affine-Norm)...")
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
                        # Parse Adapter Result
                        if "final test_loss=" in line and "mae_loss=" in line:
                             match = re.search(r"final test_loss=([0-9.]+),\s*mae_loss=([0-9.]+)", line)
                             if match:
                                 metrics["affine_norm"]["mse"] = float(match.group(1))
                                 metrics["affine_norm"]["mae"] = float(match.group(2))
                                 
                    process.wait()
                except Exception as e:
                    print(f"    Error in Phase 2: {e}")

                # Store Horizon Metrics
                print(f"  > Results for Len {pred_len}:")
                for key in ["baseline", "add", "mul", "affine", "affine_norm"]:
                    if metrics[key]["mse"] is not None:
                        horizon_metrics[key]["mse"].append(metrics[key]["mse"])
                        horizon_metrics[key]["mae"].append(metrics[key]["mae"])
                        print(f"    {key}: {metrics[key]['mse']}")
                    else:
                        print(f"    {key}: N/A")

            # ------------------------------------------------------------------
            # Calculate Averages & logging
            # ------------------------------------------------------------------
            def get_avg(metric_list):
                if not metric_list: return "N/A"
                return sum(metric_list) / len(metric_list)

            avg_baseline_mse = get_avg(horizon_metrics["baseline"]["mse"])
            avg_baseline_mae = get_avg(horizon_metrics["baseline"]["mae"])

            row = {
                "Dataset": dataset, "Model": model, "Delta": delta,
                "Avg_Baseline_MSE": avg_baseline_mse,
                "Avg_Baseline_MAE": avg_baseline_mae
            }
            
            def calc_imp(base, curr):
                if base == "N/A" or curr == "N/A": return "N/A"
                return f"{(base - curr)/base * 100:.2f}%"

            # Add, Mul, Affine, Affine-Norm
            for m in ["add", "mul", "affine", "affine_norm"]:
                avg_mse = get_avg(horizon_metrics[m]["mse"])
                avg_mae = get_avg(horizon_metrics[m]["mae"])
                
                key_mse = f"Avg_{m.replace('_', '-').title()}_MSE" 
                key_mae = f"Avg_{m.replace('_', '-').title()}_MAE"
                key_imp = f"Avg_Imp_{m.replace('_', '-').title()}(%)"
                
                row[key_mse] = avg_mse
                row[key_mae] = avg_mae
                row[key_imp] = calc_imp(avg_baseline_mse, avg_mse)
            
            # Save to CSV
            file_exists = os.path.isfile(csv_file)
            headers = ["Dataset", "Model", "Delta", "Avg_Baseline_MSE", "Avg_Baseline_MAE",
                       "Avg_Add_MSE", "Avg_Add_MAE", "Avg_Imp_Add(%)",
                       "Avg_Mul_MSE", "Avg_Mul_MAE", "Avg_Imp_Mul(%)",
                       "Avg_Affine_MSE", "Avg_Affine_MAE", "Avg_Imp_Affine(%)",
                       "Avg_Affine-Norm_MSE", "Avg_Affine-Norm_MAE", "Avg_Imp_Affine-Norm(%)"]
            
            try:
                with open(csv_file, mode='a' if file_exists else 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    if not file_exists: writer.writeheader()
                    writer.writerow(row)
                print(f"  > Saved averaged results to {csv_file}")
            except Exception as e:
                print(f"  > Error saving CSV: {e}")

    print("\n=====================================================")
    print("Comparison Experiment Finished.")

if __name__ == "__main__":
    run_comparison_experiments()
