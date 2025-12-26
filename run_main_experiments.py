import subprocess
import os
import re
import csv
import sys
import time

def run_experiment():
    # Configuration
    root_dir = os.getcwd()
    dataset_dir = os.path.join(root_dir, "datasets/")
    adapter_dir = os.path.join(root_dir, "Adapter-X+Y")
    
    # Datasets and Models
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
    models = ["iTransformer" , "Autoformer", "FreTS", "FourierGNN"]
    
    # Parameters
    delta = 0.01
    seq_len = 96
    pred_len = 96
    train_epochs = 1
    
    results = []
    
    print("Starting Main Experiments for delta-Adapter (Ada-X+Y)")
    print("=====================================================")
    
    for dataset in datasets:
        for model in models:
            print(f"----------------------------------------------------------------")
            print(f"Running Experiment: Model={model}, Dataset={dataset}, Delta={delta}")
            print(f"----------------------------------------------------------------")
            
            cmd = [
                "python", "-u", "run_xy_add.py",
                "--is_training", "1",
                "--root_path", dataset_dir,
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
                "--batch_size", "32"
            ]
            
            # Run the command
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=adapter_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                baseline_mse = None
                baseline_mae = None
                adapted_mse = None
                adapted_mae = None
                
                # Stream output and capture metrics
                for line in process.stdout:
                    print(line, end='') # Echo output to console
                    
                    # Capture Baseline Metrics from exp.test()
                    # Pattern: mse:{0:4f}, mae:{1:4f}, ...
                    if "mse:" in line and "mae:" in line and "total_loss" not in line:
                        match = re.search(r"mse:([0-9.]+),\s*mae:([0-9.]+)", line)
                        if match:
                            baseline_mse = float(match.group(1))
                            baseline_mae = float(match.group(2))
                            
                    # Capture Adapted Metrics from exp.test2()
                    # Pattern: total_loss:{0:4f}, total_mae_loss:{1:4f}
                    if "total_loss:" in line and "total_mae_loss:" in line:
                        match = re.search(r"total_loss:([0-9.]+),\s*total_mae_loss:([0-9.]+)", line)
                        if match:
                            adapted_mse = float(match.group(1))
                            adapted_mae = float(match.group(2))
                
                process.wait()
                
                if process.returncode != 0:
                    print(f"Error: Experiment failed for {model} on {dataset}")
                    continue
                    
                # Record results
                result_entry = {
                    "Dataset": dataset,
                    "Model": model,
                    "Delta": delta,
                    "Baseline_MSE": baseline_mse if baseline_mse is not None else "N/A",
                    "Baseline_MAE": baseline_mae if baseline_mae is not None else "N/A",
                    "Ada_MSE": adapted_mse if adapted_mse is not None else "N/A",
                    "Ada_MAE": adapted_mae if adapted_mae is not None else "N/A"
                }
                
                # Calculate improvement if possible
                if baseline_mse and adapted_mse:
                     improvement = (baseline_mse - adapted_mse) / baseline_mse * 100
                     result_entry["Improvement_MSE(%"] = f"{improvement:.2f}%"
                else:
                    result_entry["Improvement_MSE(%"] = "N/A"

                results.append(result_entry)
                
            except Exception as e:
                print(f"An error occurred: {e}")
                
    # Save to CSV
    csv_file = os.path.join(root_dir, "main_experiment_results.csv")
    print("\n=====================================================")
    print(f"Saving results to {csv_file}")
    
    fieldnames = ["Dataset", "Model", "Delta", "Baseline_MSE", "Baseline_MAE", "Ada_MSE", "Ada_MAE", "Improvement_MSE(%"]
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print("Done.")

if __name__ == "__main__":
    run_experiment()
