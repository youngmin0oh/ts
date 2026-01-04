import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def visualize_results(csv_path):
    # Load data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return

    # Columns to interest
    imp_cols = {
        'Imp_Add(%)': 'Add',
        'Imp_Mul(%)': 'Mul',
        'Imp_Norm(%)': 'Norm',
        'Imp_Affine(%)': 'Affine',
        'Imp_Affine-Norm(%)': 'Our Method'
    }

    # Clean percentage columns
    for col in imp_cols.keys():
        if col in df.columns:
            # Remove '%' and convert to float
            df[col] = df[col].astype(str).str.rstrip('%').astype(float)
        else:
            print(f"Warning: Column {col} not found in CSV.")

    # Prepare data for plotting
    plot_data = []
    
    # We want to group by Dataset and Model, or just Dataset
    # Let's melt the dataframe to have a long format: Dataset, Model, Method, Improvement
    
    melted_df = df.melt(id_vars=['Dataset', 'Model'], 
                        value_vars=[c for c in imp_cols.keys() if c in df.columns],
                        var_name='Method_Col', 
                        value_name='Improvement')
    
    # Map column names to display names
    melted_df['Method'] = melted_df['Method_Col'].map(imp_cols)

    # Set style
    sns.set_theme(style="whitegrid")
    
    # Create the plot
    # We'll plot Average Improvement by Method across all Datasets and Models first
    # Or grouped by Dataset
    
    # Figure 1: Improvement by Dataset (averaged over models)
    plt.figure(figsize=(14, 8))
    sns.barplot(data=melted_df, x='Dataset', y='Improvement', hue='Method', errorbar=None, palette="viridis")
    
    plt.title('Average Improvement by Dataset (SeqLen 96)', fontsize=16)
    plt.ylabel('Improvement (%)', fontsize=14)
    plt.xlabel('Dataset', fontsize=14)
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_filename = os.path.splitext(os.path.basename(csv_path))[0] + '.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Saved plot to {output_filename}")
    plt.close()

    # Figure 2: Detailed Improvement by Dataset and Model
    # Since there might be many bars, let's make a FacetGrid or a larger plot
    # Or maybe just print the averages to console for verification
    
    avg_imp = melted_df.groupby('Method')['Improvement'].mean()
    print("\nOverall Average Improvement by Method:")
    print(avg_imp)

if __name__ == "__main__":
    csv_file = "comparison_experiment_results_seqlen_96.csv"
    visualize_results(csv_file)
