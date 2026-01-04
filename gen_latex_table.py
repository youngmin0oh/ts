import re

latex_input = r"""
\multirow{4}{*}{ETTh1} 
 & iTransformer & 0.465 & 0.469 & 0.462 & 0.77\% & 0.462 & 0.73\% & 0.459 & 1.29\% & \textbf{0.456} & \textbf{2.00\%} \\
 & Autoformer & 0.613 & 0.560 & 0.608 & 0.83\% & 0.611 & 0.42\% & 0.602 & 1.90\% & \textbf{0.585} & \textbf{4.61\%} \\
 & FreTS & 0.486 & 0.486 & 0.482 & 0.64\% & 0.484 & 0.36\% & 0.482 & 0.73\% & \textbf{0.473} & \textbf{2.54\%} \\
 & FourierGNN & 0.916 & 0.697 & 0.903 & 1.46\% & 0.910 & 0.72\% & 0.871 & 4.98\% & \textbf{0.825} & \textbf{10.02\%} \\
\midrule
\multirow{4}{*}{ETTh2} 
 & iTransformer & 0.191 & 0.299 & 0.190 & 0.76\% & 0.190 & 0.71\% & 0.192 & -0.33\% & \textbf{0.189} & \textbf{1.18\%} \\
 & Autoformer & 0.210 & 0.315 & 0.208 & 1.22\% & 0.208 & 0.79\% & 0.209 & 0.41\% & \textbf{0.204} & \textbf{2.93\%} \\
 & FreTS & 0.193 & 0.304 & 0.192 & 0.52\% & 0.193 & 0.05\% & 0.196 & -1.51\% & \textbf{0.189} & \textbf{1.92\%} \\
 & FourierGNN & 0.321 & 0.406 & 0.315 & 1.91\% & 0.317 & 1.35\% & 0.289 & 9.80\% & \textbf{0.281} & \textbf{12.48\%} \\
\midrule
\multirow{4}{*}{ETTm1} 
 & iTransformer & 0.376 & 0.409 & 0.374 & 0.45\% & 0.375 & 0.24\% & 0.373 & 0.88\% & \textbf{0.374} & \textbf{0.64\%} \\
 & Autoformer & 0.585 & 0.544 & 0.571 & 2.37\% & 0.571 & 2.39\% & 0.529 & 9.45\% & \textbf{0.507} & \textbf{13.22\%} \\
 & FreTS & 0.374 & 0.407 & 0.370 & 0.84\% & 0.373 & 0.25\% & 0.361 & 3.25\% & \textbf{0.368} & \textbf{1.59\%} \\
 & FourierGNN & 0.534 & 0.512 & 0.524 & 1.81\% & 0.528 & 1.21\% & 0.480 & 10.12\% & \textbf{0.451} & \textbf{15.57\%} \\
\midrule
\multirow{4}{*}{ETTm2} 
 & iTransformer & 0.124 & 0.240 & 0.121 & 1.87\% & 0.122 & 1.14\% & 0.119 & 3.57\% & \textbf{0.114} & \textbf{7.38\%} \\
 & Autoformer & 0.153 & 0.273 & 0.151 & 1.41\% & 0.151 & 0.95\% & 0.147 & 4.03\% & \textbf{0.126} & \textbf{17.51\%} \\
 & FreTS & 0.128 & 0.248 & 0.126 & 1.44\% & 0.127 & 1.20\% & 0.123 & 4.17\% & \textbf{0.115} & \textbf{10.18\%} \\
 & FourierGNN & 0.152 & 0.277 & 0.149 & 2.00\% & 0.151 & 0.82\% & 0.149 & 2.46\% & \textbf{0.125} & \textbf{17.89\%} \\
"""

def parse_imp(val_str):
    # Remove \textbf, %, and whitespace
    clean = val_str.replace(r'\textbf{', '').replace('}', '').replace(r'\%', '').strip()
    return float(clean)

lines = latex_input.strip().split('\n')
output_lines = []

for line in lines:
    if r'\midrule' in line or r'\multirow' in line:
        output_lines.append(line)
        continue
    
    parts = line.split('&')
    if len(parts) < 12:
        output_lines.append(line)
        continue
        
    # Indices:
    # 0: &
    # 1: Model
    # 2: Base MSE
    # 3: Base MAE
    # 4: Add MSE
    # 5: Add Imp
    # 6: Mul MSE
    # 7: Mul Imp
    # 8: Affine MSE
    # 9: Affine Imp
    # 10: Ours MSE
    # 11: Ours Imp (ends with \\)
    
    # Extract values
    add_imp_val = parse_imp(parts[5])
    mul_imp_val = parse_imp(parts[7])
    ours_imp_val = parse_imp(parts[11].replace(r'\\', ''))
    
    add_diff = ours_imp_val - add_imp_val
    mul_diff = ours_imp_val - mul_imp_val
    
    # Format diff strings with color
    # Using ForestGreen or similar
    def format_diff(diff):
        sign = "+" if diff >= 0 else ""
        return r"\tiny{\textcolor{teal}{(%s%.2f\%%)}}" % (sign, diff)

    new_add_imp = parts[5].strip() + " " + format_diff(add_diff)
    new_mul_imp = parts[7].strip() + " " + format_diff(mul_diff)
    
    # Construct new line
    # Remove Affine parts (8, 9)
    # New parts: 0, 1, 2, 3, 4, new_add_imp, 6, new_mul_imp, 10, 11
    
    new_parts = [
        parts[0],
        parts[1],
        parts[2],
        parts[3],
        parts[4],
        new_add_imp,
        parts[6],
        new_mul_imp,
        # skip 8, 9
        parts[10],
        parts[11]
    ]
    
    output_lines.append(" & ".join(new_parts))

# Header generation
header = r"""
\begin{table*}[htbp]
\caption{Comparative Analysis of Adaptation Methods. "Imp" denotes improvement over baseline. Values in () indicate the additional improvement of our method over the respective adapter.}
\begin{center}
\begin{tabular}{llcccccccc}
\toprule
\multirow{2}{*}{Dataset} & \multirow{2}{*}{Model} & \multicolumn{2}{c}{Baseline} & \multicolumn{2}{c}{Add-Adapter} & \multicolumn{2}{c}{Mul-Adapter} & \multicolumn{2}{c}{\textbf{Affine-Norm (Ours)}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}
 & & MSE & MAE & MSE & Imp\% & MSE & Imp\% & \textbf{MSE} & \textbf{Imp\%} \\
\midrule
"""

footer = r"""\bottomrule
\end{tabular}
\label{tab:main_results}
\end{center}
\end{table*}
"""

print(header)
for line in output_lines:
    print(line)
print(footer)
