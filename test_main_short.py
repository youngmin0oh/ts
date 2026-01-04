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
    
    # Datasets and Models - REDUCED SCOPE
    datasets = ["ETTh2"]
    models = ["iTransformer"]
    
    # Parameters
    delta = 0.01
    seq_len = 96
    pred_len = 96
    train_epochs = 1
    
    results = []
    
    # REDUCED HORIZONS for testing
    pred_lens = [96, 192]
    
    print("Starting Main Experiments for delta-Adapter (Ada-X+Y) - Multi-Horizon TEST")
    print("=====================================================")
    
    for dataset in datasets:
        for model in models:
            
            # Storage for horizon results
            horizon_metrics = {
                "baseline": {"mse": [], "mae": []},
                "add": {"mse": [], "mae": []},
                "mul": {"mse": [], "mae": []},
                "affine": {"mse": [], "mae": []}
            }
            
            for pred_len in pred_lens:
                print(f"----------------------------------------------------------------")
                print(f"Model={model}, Dataset={dataset}, Delta={delta}, PredLen={pred_len}")
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
                    
                    # Dictionary to store metrics for each mode (Current Horizon)
                    adapter_results = {
                        "add": {"mse": None, "mae": None},
                        "mul": {"mse": None, "mae": None},
                        "affine": {"mse": None, "mae": None}
                    }
                    
                    current_mode = None

                    # Stream output and capture metrics
                    for line in process.stdout:
                        # print(line, end='') # Supress output for test cleanliness
                        
                        # Detect current mode being run
                        if "Running Adapter Mode:" in line:
                            if "add" in line: current_mode = "add"
                            elif "mul" in line: current_mode = "mul"
                            elif "affine" in line: current_mode = "affine"
                        
                        # Capture Baseline Metrics from exp.test()
                        if "mse:" in line and "mae:" in line and "total_loss" not in line and "final test_loss" not in line and baseline_mse is None:
                            match = re.search(r"mse:([0-9.]+),\s*mae:([0-9.]+)", line)
                            if match:
                                baseline_mse = float(match.group(1))
                                baseline_mae = float(match.group(2))
                                
                        # Capture Adapted Metrics from exp.test2()
                        if "final test_loss=" in line and "mae_loss=" in line:
                            mode_in_line_match = re.search(r"\[(add|mul|affine)\]", line)
                            parsed_mode = mode_in_line_match.group(1) if mode_in_line_match else current_mode
                            
                            if parsed_mode:
                                match = re.search(r"test_loss=([0-9.]+),\s*mae_loss=([0-9.]+)", line)
                                if match:
                                    adapter_results[parsed_mode]["mse"] = float(match.group(1))
                                    adapter_results[parsed_mode]["mae"] = float(match.group(2))
                    
                    process.wait()
                    
                    if process.returncode != 0:
                        print(f"Error: Experiment failed for {model} on {dataset} (Len {pred_len})")
                        continue
                    
                    # Store results for this horizon
                    if baseline_mse is not None:
                        horizon_metrics["baseline"]["mse"].append(baseline_mse)
                        horizon_metrics["baseline"]["mae"].append(baseline_mae)
                        print(f"  Got results for PredLen={pred_len}: Baseline_MSE={baseline_mse}")
                    else:
                        print(f"  Failed to parse baseline for {pred_len}")
                    
                    for m in ["add", "mul", "affine"]:
                        if adapter_results[m]["mse"] is not None:
                            horizon_metrics[m]["mse"].append(adapter_results[m]["mse"])
                            horizon_metrics[m]["mae"].append(adapter_results[m]["mae"])
                            
                except Exception as e:
                    print(f"An error occurred: {e}")

            # --- End of Horizon Loop ---
            
            # Calculate Averages over horizons
            def get_avg(metric_list):
                if not metric_list: return "N/A"
                return sum(metric_list) / len(metric_list)

            avg_baseline_mse = get_avg(horizon_metrics["baseline"]["mse"])
            avg_baseline_mae = get_avg(horizon_metrics["baseline"]["mae"])
            
            print(f"Average Baseline MSE over {pred_lens}: {avg_baseline_mse}")

            # Record Average Results
            result_entry = {
                "Dataset": dataset,
                "Model": model,
                "Delta": delta,
                "Avg_Baseline_MSE": avg_baseline_mse,
                "Avg_Baseline_MAE": avg_baseline_mae
            }
            
            # Process Adapters
            for mode in ["Add", "Mul", "Affine"]:
                mode_key = mode.lower()
                avg_mse = get_avg(horizon_metrics[mode_key]["mse"])
                avg_mae = get_avg(horizon_metrics[mode_key]["mae"])
                
                result_entry[f"Avg_{mode}_MSE"] = avg_mse
                result_entry[f"Avg_{mode}_MAE"] = avg_mae
                
                # Calculate Improvement based on AVERAGES
                if avg_baseline_mse != "N/A" and avg_mse != "N/A":
                    imp = (avg_baseline_mse - avg_mse) / avg_baseline_mse * 100
                    result_entry[f"Avg_Imp_{mode}(%)"] = f"{imp:.2f}%"
                else:
                    result_entry[f"Avg_Imp_{mode}(%)"] = "N/A"

            # Save to CSV (Incremental)
            csv_file = os.path.join(root_dir, "test_results_avg.csv")
            file_exists = os.path.isfile(csv_file)
            
            fieldnames = ["Dataset", "Model", "Delta", "Avg_Baseline_MSE", "Avg_Baseline_MAE", 
                          "Avg_Add_MSE", "Avg_Add_MAE", "Avg_Imp_Add(%)",
                          "Avg_Mul_MSE", "Avg_Mul_MAE", "Avg_Imp_Mul(%)",
                          "Avg_Affine_MSE", "Avg_Affine_MAE", "Avg_Imp_Affine(%)"]
            
            try:
                 with open(csv_file, mode='a' if file_exists else 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(result_entry)
                 print(f"Averaged Results saved to {csv_file}")
            except Exception as e:
                 print(f"Error saving to CSV: {e}")
                
    print("\n=====================================================")
    print("Test Experiment finished.")

if __name__ == "__main__":
    run_experiment()
