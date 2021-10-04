import numpy as np
import pandas as pd
import os
import pathlib as pl
import phimanip as pm



def normalize_2d(matrix):
    abs = np.abs(matrix)
    max = np.max(abs)
    norm = matrix/max
    sized = norm*np.pi/2
    return sized
path = pl.Path("C:/Users/fwessel/Desktop/2m_lens")
values  = pd.read_csv(os.path.join(path,"phi_avg_40.csv")),\
          pd.read_csv(os.path.join(path, "phi_bg_avg_40.csv"))

count = 0
for file in values:
    norms = normalize_2d(file)
    if count == 0:
        norms.to_csv(os.path.join(path, "sample_normalized.csv"), index =None)
        count+=1
    else:
        norms.to_csv(os.path.join(path, "bg_normalized.csv"), index =None)


