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
                "python", "-u", "run_xy_combined.py",
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
                "--batch_size", "32",
                "--adapter_mode", "all"  # Run Add, Mul, and Affine
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
                
                # Dictionary to store metrics for each mode
                adapter_results = {
                    "add": {"mse": None, "mae": None},
                    "mul": {"mse": None, "mae": None},
                    "affine": {"mse": None, "mae": None}
                }
                
                current_mode = None

                # Stream output and capture metrics
                for line in process.stdout:
                    print(line, end='') # Echo output to console
                    
                    # Detect current mode being run
                    if "Running Adapter Mode:" in line:
                        if "add" in line: current_mode = "add"
                        elif "mul" in line: current_mode = "mul"
                        elif "affine" in line: current_mode = "affine"
                    
                    # Capture Baseline Metrics from exp.test()
                    # Pattern: mse:{0:4f}, mae:{1:4f}, ...
                    if "mse:" in line and "mae:" in line and "total_loss" not in line and "final test_loss" not in line and baseline_mse is None:
                        match = re.search(r"mse:([0-9.]+),\s*mae:([0-9.]+)", line)
                        if match:
                            baseline_mse = float(match.group(1))
                            baseline_mae = float(match.group(2))
                            
                    # Capture Adapted Metrics from exp.test2()
                    # Pattern: [mode] final test_loss={test_loss:.4f}, mae_loss={mae_loss:.4f}
                    if "final test_loss=" in line and "mae_loss=" in line:
                        # Attempt to parse mode from line if present, otherwise rely on context
                        mode_in_line_match = re.search(r"\[(add|mul|affine)\]", line)
                        parsed_mode = mode_in_line_match.group(1) if mode_in_line_match else current_mode
                        
                        if parsed_mode:
                            match = re.search(r"test_loss=([0-9.]+),\s*mae_loss=([0-9.]+)", line)
                            if match:
                                adapter_results[parsed_mode]["mse"] = float(match.group(1))
                                adapter_results[parsed_mode]["mae"] = float(match.group(2))
                
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
                    
                    "Add_MSE": adapter_results["add"]["mse"] if adapter_results["add"]["mse"] is not None else "N/A",
                    "Add_MAE": adapter_results["add"]["mae"] if adapter_results["add"]["mae"] is not None else "N/A",
                    
                    "Mul_MSE": adapter_results["mul"]["mse"] if adapter_results["mul"]["mse"] is not None else "N/A",
                    "Mul_MAE": adapter_results["mul"]["mae"] if adapter_results["mul"]["mae"] is not None else "N/A",
                    
                    "Affine_MSE": adapter_results["affine"]["mse"] if adapter_results["affine"]["mse"] is not None else "N/A",
                    "Affine_MAE": adapter_results["affine"]["mae"] if adapter_results["affine"]["mae"] is not None else "N/A",
                }
                
                # Calculate improvement for best adapter? Or just record raw?
                # Let's record improvements for each
                for mode in ["Add", "Mul", "Affine"]:
                     if baseline_mse and result_entry[f"{mode}_MSE"] != "N/A":
                         imp = (baseline_mse - result_entry[f"{mode}_MSE"]) / baseline_mse * 100
                         result_entry[f"Imp_{mode}(%)"] = f"{imp:.2f}%"
                     else:
                         result_entry[f"Imp_{mode}(%)"] = "N/A"

                # Save to CSV (Incremental)
                csv_file = os.path.join(root_dir, "main_experiment_results_add_mul_affine.csv")
                file_exists = os.path.isfile(csv_file)
                
                fieldnames = ["Dataset", "Model", "Delta", "Baseline_MSE", "Baseline_MAE", 
                              "Add_MSE", "Add_MAE", "Imp_Add(%)",
                              "Mul_MSE", "Mul_MAE", "Imp_Mul(%)",
                              "Affine_MSE", "Affine_MAE", "Imp_Affine(%)"]
                
                try:
                     with open(csv_file, mode='a' if file_exists else 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                        writer.writerow(result_entry)
                     print(f"Results saved to {csv_file}")
                except Exception as e:
                     print(f"Error saving to CSV: {e}")

                results.append(result_entry)
                
            except Exception as e:
                print(f"An error occurred: {e}")
                
    print("\n=====================================================")
    print("Experiment finished.")

if __name__ == "__main__":
    run_experiment()
