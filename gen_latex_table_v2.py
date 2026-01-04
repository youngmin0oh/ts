import pandas as pd
import numpy as np

def format_imp(base, current):
    # Calculate improvement %: (base - current) / base * 100
    if base == 0: return "0.00"
    imp = (base - current) / base * 100
    return f"{imp:.2f}\%"

def format_relative_imp(adapter, ours):
    # Improvement of Ours over Adapter: (Adapter - Ours) / Adapter * 100
    if adapter == 0: return ""
    imp = (adapter - ours) / adapter * 100
    sign = "+" if imp >= 0 else ""
    # Use teal color for existing command
    return f"\\tiny{{\\textcolor{{teal}}{{({sign}{imp:.2f}\\%)}}}}"

def main():
    csv_path = "comparison_experiment_results_seqlen_96.csv"
    df = pd.read_csv(csv_path)
    
    # Filter datasets and models if needed (assuming all in CSV are needed)
    # The snippet had ETTh1, ETTh2, ETTm1, ETTm2
    # And models: iTransformer, Autoformer, FreTS, FourierGNN
    
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
    models = ['iTransformer', 'Autoformer', 'FreTS', 'FourierGNN']
    
    # Header
    print(r"\begin{table*}[htbp]")
    print(r"\caption{Comparative Analysis of Adaptation Methods including MAE. \"Imp\" denotes improvement over baseline. Values in () indicate the additional improvement of our method over the respective adapter.}")
    print(r"\begin{center}")
    # Columns: Dataset, Model, Metric, Base, Add, Mul, Ours
    print(r"\begin{tabular}{lllcccc}")
    print(r"\toprule")
    print(r"Dataset & Model & Metric & Baseline & Add-Adapter & Mul-Adapter & \textbf{Affine-Norm (Ours)} \\")
    print(r" & & & Value & Value (Imp\%) & Value (Imp\%) & \textbf{Value (Imp\%)} \\")
    print(r"\midrule")
    
    for dataset in datasets:
        ds_df = df[df['Dataset'] == dataset]
        if ds_df.empty: continue
        
        print(f"\\multirow{{8}}{{*}}{{{dataset}}} ")
        
        for model in models:
            row = ds_df[ds_df['Model'] == model]
            if row.empty: continue
            row = row.iloc[0]
            
            # MSE Row
            base_mse = row['Baseline_MSE']
            add_mse = row['Add_MSE']
            mul_mse = row['Mul_MSE']
            ours_mse = row['Affine-Norm_MSE']
            
            # MSE Strings
            add_mse_str = f"{add_mse:.3f} ({format_imp(base_mse, add_mse)}) {format_relative_imp(add_mse, ours_mse)}"
            mul_mse_str = f"{mul_mse:.3f} ({format_imp(base_mse, mul_mse)}) {format_relative_imp(mul_mse, ours_mse)}"
            ours_mse_str = f"\\textbf{{{ours_mse:.3f} ({format_imp(base_mse, ours_mse)})}}"
            
            print(f" & \\multirow{{2}}{{*}}{{{model}}} & MSE & {base_mse:.3f} & {add_mse_str} & {mul_mse_str} & {ours_mse_str} \\\\")
            
            # MAE Row
            base_mae = row['Baseline_MAE']
            add_mae = row['Add_MAE']
            mul_mae = row['Mul_MAE']
            ours_mae = row['Affine-Norm_MAE']
            
            # MAE Strings
            add_mae_str = f"{add_mae:.3f} ({format_imp(base_mae, add_mae)}) {format_relative_imp(add_mae, ours_mae)}"
            mul_mae_str = f"{mul_mae:.3f} ({format_imp(base_mae, mul_mae)}) {format_relative_imp(mul_mae, ours_mae)}"
            ours_mae_str = f"\\textbf{{{ours_mae:.3f} ({format_imp(base_mae, ours_mae)})}}"
            
            print(f" & & MAE & {base_mae:.3f} & {add_mae_str} & {mul_mae_str} & {ours_mae_str} \\\\")
            
            # Add small separator if not last model?? No standard style is fine.
        
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\label{tab:main_results_mae}")
    print(r"\end{center}")
    print(r"\end{table*}")

if __name__ == "__main__":
    main()
