from tkinter.filedialog import askopenfilename
from scipy import optimize

import numpy as np
import pandas as pd
from lmfit import Model

# These two lines are error handling
old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)


def theoretical_calibration_curve(x, alpha, v, p, q):
    numerator = alpha + (v * np.sin((p*x) + q))
    denominator = 1 + (alpha * v * np.sin((p*x) + q))
    return (numerator/denominator)


def characterize():

    filename_r_matrices_stats = askopenfilename(
        title='Pick a r_matrices_stats_file')  # show an "Open" dialog box and return the path to the selected file

    df = pd.read_csv(filepath_or_buffer=filename_r_matrices_stats)

    frame_index = df.loc[:, 'Frame'].values
    r_values = df.loc[:, 'Avg_R'].values


    min_frame = int(df.loc[[df['Avg_R'].idxmin()], :].iloc[0]["Frame"])
    max_frame = int(df.loc[[df['Avg_R'].idxmax()], :].iloc[0]["Frame"])


    #model = Model(theoretical_calibration_curve, independent_vars=['x'])
    arb_alpha = 0.1
    arb_v = .96
    arb_p = .068
    arb_q = 5.9                 # uses data to make the best guess at a frequency, every calibration curve will have a different frequency depending on the number of frames taken during the curve
    arb_c = 0.2

    print("Starting Guesses for Algo.")
    print("arb_alpha: ", arb_alpha)
    print("arb_v: ", arb_v)
    print("arb_q: ", arb_q)
    print("arb_p: ", arb_p)
    print("arb_c: ", arb_c)

    #params, params_covariance = optimize.curve_fit(theoretical_calibration_curve, frame_index, r_values,
                                                   #p0=[arb_alpha, arb_v, arb_p, arb_q, arb_c])

    #result = model.fit(r_values, x=frame_index, alpha=arb_alpha, v=arb_v, p=arb_p, q=arb_q)
    #value_alpha = float(result.values['alpha'])
    #value_v = float(result.values['v'])

    params, params_covariance = optimize.curve_fit(theoretical_calibration_curve, frame_index, r_values,
                                                   p0=[arb_alpha, arb_v, arb_p, arb_q])
    trunc1 = f"{params[0]:.3f}"  # alpha
    trunc2 = f"{params[1]:.3f}"  # v


    print("Parametry Est. Results")
    print("alpha: ", params[0])
    print("v: ", params[1])
    print("q: ", params[3])
    print("p: ", params[2])
    #print("c: ", params[4])
    #print("alpha:",trunc1,"v",trunc2)
    #result.plot()
    #plt.show()
    #print(params)
    return filename_r_matrices_stats, params[0], params[1], min_frame, max_frame
