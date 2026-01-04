import os
import pandas as pd
import numpy as np
dirpath = "datasets"
for f in os.listdir(dirpath):
    df = pd.read_csv(os.path.join(dirpath, f))
    print(f, df.shape)
