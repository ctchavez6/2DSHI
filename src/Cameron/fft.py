import numpy
import numpy as np
import pandas as pd
import os
import pathlib as pl
from plotly.subplots import make_subplots
import plotly.graph_objects as go




def fft(matrix):
    out1 = np.fft.rfft2(matrix)
    out3 = abs(out1)
    out2 = pd.DataFrame(out3)
    out = np.clip(out2,0,1000)
    return out

path = pl.Path("D:/2021_08_31__10_53/non_averaged")
values  = pd.read_csv(os.path.join(path,"phi.csv"), header = None),\
          pd.read_csv(os.path.join(path, "phi_bg.csv"),header =None)


first = fft(values[0])

second = fft(values[1])


x = np.linspace(-500,500,1001)
y = np.linspace(-500,500,1001)

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]]
    )

fig.add_trace(
    go.Surface(z=first, colorscale='YlOrRd', showscale=False), row=1, col=1)
fig.add_trace(
    go.Surface(z=second, colorscale='YlOrRd', showscale=False), row=1, col=2)
fig.show()




    # if count == 0:
    #     norms.to_csv(os.path.join(path, "sample_normalized.csv"), index =None)
    #     count+=1
    # else:
    #     norms.to_csv(os.path.join(path, "bg_normalized.csv"), index =None)