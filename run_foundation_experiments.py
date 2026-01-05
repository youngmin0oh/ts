import subprocess
import os
import re
import csv
import sys
import time

def run_foundation_experiments():
    # Configuration
    root_dir = os.getcwd()
    script_original = "Adapter-X+Y/run_xy_combined.py"
    script_norm = "Adapter-X+Y/run_xy_combined_norm.py"
    
    # Models and Datasets (Table 1 reproduction)
    models = ["Sundial", "TTM"]
    datasets = ["ELC", "ETTm2", "Exchange", "Traffic", "Weather"]
    
    # Parameters
    seq_len = 512 # Foundation models fixed context
    pred_len = 96
    delta = 0.01 # Global delta as per user feedback
    
    print("Starting Expanded Table 1 Reproduction: Foundation Models with All Adapter Modes")
    print("============================================================================")
    
    csv_file = os.path.join(root_dir, "table1_results_expanded.csv")
    headers = ["Dataset", "Model", "Delta",
               "Baseline_MSE", "Baseline_MAE", 
               "Add_MSE", "Add_MAE", "Imp_Add(%)",
               "Mul_MSE", "Mul_MAE", "Imp_Mul(%)",
               "Affine_MSE", "Affine_MAE", "Imp_Affine(%)",
               "Norm_MSE", "Norm_MAE", "Imp_Norm(%)",
               "OurMethod_MSE", "OurMethod_MAE", "Imp_OurMethod(%)"]
    
    # Initialize CSV if not exists or write header
    file_exists = os.path.isfile(csv_file)
    if not file_exists:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    for model in models:
        for dataset in datasets:
            print(f"\nProcessing Model: {model}, Dataset: {dataset}, Delta: {delta}")
            print("----------------------------------------------------------------")
            
            # Dictionary to store metrics for all modes
            metrics = {
                "baseline": {"mse": None, "mae": None},
                "add": {"mse": None, "mae": None},
                "mul": {"mse": None, "mae": None},
                "affine": {"mse": None, "mae": None},
                "norm": {"mse": None, "mae": None},
                "our_method": {"mse": None, "mae": None}
            }
            
            # Base arguments for standard script
            base_args_orig = [
                "python", "-u", script_original,
                "--is_training", "1",
                "--root_path", "./datasets/",
                "--data_path", f"{dataset}.csv",
                "--model_id", f"{dataset}_{model}_{pred_len}", 
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
                "--train_epochs", "0", 
                "--batch_size", "1", # Low batch size for foundation models
                "--num_episodes", "1", # Single pass adaptation for speed
                "--checkpoints", "Adapter-X+Y/AdaCali/checkpoints/",
                "--gpu", "0"
            ]

            # ------------------------------------------------------------------
            # Phase 1: Run Standard Adapters (Add, Mul, Affine) + Baseline
            # ------------------------------------------------------------------
            print("  > Phase 1: Running Standard Adapters (Add, Mul, Affine)...")
            cmd_orig = base_args_orig + ["--adapter_mode", "all", "--adapter_target", "y"]
            
            try:
                process = subprocess.Popen(cmd_orig, cwd=root_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                current_mode_ctx = None
                for line in process.stdout:
                    # Optional: print(line, end="") # For live debugging
                    
                    # Parse Baseline
                    if "mse:" in line and "mae:" in line and "total_loss" not in line and metrics["baseline"]["mse"] is None:
                        match = re.search(r"mse:([0-9.]+),\s*mae:([0-9.]+)", line)
                        if match:
                            metrics["baseline"]["mse"] = float(match.group(1))
                            metrics["baseline"]["mae"] = float(match.group(2))
                    
                    # Detect Mode Context
                    if "Running Adapter Mode:" in line:
                        if "add" in line: current_mode_ctx = "add"
                        elif "mul" in line: current_mode_ctx = "mul"
                        elif "affine" in line: current_mode_ctx = "affine"

                    # Parse Adapted Metrics
                    if "final test_loss=" in line and "mae_loss=" in line:
                        match = re.search(r"final test_loss=([0-9.]+),\s*mae_loss=([0-9.]+)", line)
                        if match and current_mode_ctx:
                             metrics[current_mode_ctx]["mse"] = float(match.group(1))
                             metrics[current_mode_ctx]["mae"] = float(match.group(2))
                process.wait()
            except Exception as e:
                print(f"    Error in Phase 1: {e}")

            # ------------------------------------------------------------------
            # Phase 2: Run Normalization Adapters (Norm, Our Method)
            # ------------------------------------------------------------------
            print("  > Phase 2: Running Normalization Adapters (Norm, Our Method)...")
            
            phase2_modes = [("norm", "norm"), ("our_method", "affine")] # (metric_key, script_mode_arg)
            
            for metric_key, mode_arg in phase2_modes:
                print(f"    >> Mode: {metric_key}")
                cmd_norm = list(base_args_orig)
                cmd_norm[2] = script_norm # Swap script path
                cmd_norm += ["--adapter_mode", mode_arg, "--adapter_target", "y"]
                
                try:
                    process = subprocess.Popen(cmd_norm, cwd=root_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                    for line in process.stdout:
                        if "final test_loss=" in line and "mae_loss=" in line:
                            match = re.search(r"final test_loss=([0-9.]+),\s*mae_loss=([0-9.]+)", line)
                            if match:
                                 metrics[metric_key]["mse"] = float(match.group(1))
                                 metrics[metric_key]["mae"] = float(match.group(2))
                    process.wait()
                except Exception as e:
                    print(f"    Error in Phase 2 ({metric_key}): {e}")

            # ------------------------------------------------------------------
            # Calculate Results & Reporting
            # ------------------------------------------------------------------
            def safe_format(val): return f"{val:.4f}" if val is not None else "N/A"
            def calc_imp(base, curr):
                if base is None or curr is None: return "N/A"
                if base == 0: return "0.00%"
                return f"{(base - curr)/base * 100:.2f}%"

            row = {
                "Dataset": dataset, "Model": model, "Delta": delta,
                "Baseline_MSE": safe_format(metrics["baseline"]["mse"]),
                "Baseline_MAE": safe_format(metrics["baseline"]["mae"])
            }
            
            for m in ["add", "mul", "affine", "norm", "our_method"]:
                mse_key = "OurMethod_MSE" if m == "our_method" else f"{m.title()}_MSE"
                mae_key = "OurMethod_MAE" if m == "our_method" else f"{m.title()}_MAE"
                imp_key = "Imp_OurMethod(%)" if m == "our_method" else f"Imp_{m.title()}(%)"
                
                row[mse_key] = safe_format(metrics[m]["mse"])
                row[mae_key] = safe_format(metrics[m]["mae"])
                row[imp_key] = calc_imp(metrics["baseline"]["mse"], metrics[m]["mse"])
            
            print(f"Results recorded for {model}/{dataset}")
            
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(row)
                
    print(f"\nCompleted. Results saved to {csv_file}")

if __name__ == "__main__":
    run_foundation_experiments()
