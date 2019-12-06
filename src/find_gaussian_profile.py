import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from astropy import modeling
from scipy.optimize import curve_fit
from lmfit import Model

def gaussian_distribution(x, mu, sigma, coefficient):
    """
    Returns values for a gaussian distribution based on the input array.
    Args:
        x: Array of x values.
        mu: Mean/Median/Mode/
        sigma: Standard Deviation
        coefficient: Amplitude of gaussian distribution
    Raises:
        Exception: If any of the input parameters are not of the specified type.
    Returns:
        distribution: An array of y-values that correspond to the input parameters.
    """
    #distribution = coefficient*np.exp(-(x-mu)**2/(2*sigma**2))
    return coefficient * np.exp(-(x-mu)**2/(2.0*sigma**2))
    #return distribution


def read_image_from_file(image_path, file_bit_depth=16, original_bit_depth=12):
    """
    Given a file path, returns the corresponding image array. You may also specified the original/intended bit depth
    of said image.

    Args:
        image_path: Path to the image you want to read in.
        original_bit_depth: An integer
        file_bit_depth: An integer
    Raises:
        Exception: Nothing so far.
    Returns:
        The image array in it's intended bit depth.
    """
    return np.asarray(cv2.imread(image_path, -1)) / (2**(file_bit_depth-original_bit_depth))


def get_maximum_pixel_intensity(image_array):
    """
    Takes an image array (2D grayscale), flattens it into 1D, and finds the maximum.

    Args:
        image_array: Image as a 2D numpy array.
    Returns:
        The image intensity maximum (assuming grayscale) as an int.
    """
    return max(image_array.flatten())


def get_coordinates_of_maximum(image):
    """
    Takes a 2D (grayscale) image array and finds the coordinates of the pixel with the maximum intensity value. If
    multiple pixels share the same maximum intensity value, returns centroid of those coordinates.

    Args:
        image: Image as a 2D numpy array.
    Returns:
        The image intensity maximum (assuming grayscale) as an int.
    """
    maximum_intensity_value = get_maximum_pixel_intensity(image)
    x_coordinates = np.where(image == maximum_intensity_value)[1]
    y_coordinates = np.where(image == maximum_intensity_value)[0]
    if len(x_coordinates) > 1 or len(y_coordinates) > 1:
        return [(int(np.mean(x_coordinates)), int(np.mean(y_coordinates)))][0]
    return [(i, j) for i, j in zip(x_coordinates, y_coordinates)][0]


def crop_surrounding_noise(image, maximum_coords, upper_limit_noise=10):
    """
    Given an image and the coordinates of the maximum intensity, crops out noise surrounding signal.

    Args:
        image:
        maximum_coords:
        upper_limit_noise: Pixels with intensity values less than or equal to this noise limit will be cropped out from
        the image.
    Returns:
        np.ndarray: Newly cropped image
    """

    horizontal_center, vertical_center = maximum_coords[0], maximum_coords[1]
    horizontal_lineout = np.asarray(image[vertical_center:vertical_center+1, :])[0]
    vertical_lineout = image[:, horizontal_center:horizontal_center+1].flatten()
    index_of_1st_pixel_above_noise_limit_h = next(x[0] for x in enumerate(horizontal_lineout) if x[1] > upper_limit_noise)
    index_of_last_pixel_above_noise_limit_flo_h = len(horizontal_lineout) - next(x[0] for x in enumerate(reversed(horizontal_lineout)) if x[1] > upper_limit_noise) - 1

    index_of_1st_pixel_above_noise_limit_v = next(x[0] for x in enumerate(vertical_lineout) if x[1] > upper_limit_noise)
    index_of_last_pixel_above_noise_limit_flo_v = len(vertical_lineout) - next(x[0] for x in enumerate(reversed(vertical_lineout)) if x[1] > upper_limit_noise) - 1

    fig_a, ax = plt.subplots()
    ax.title.set_text('Original Image')
    ax.imshow(image)
    fig_a.savefig("Original_Image.png")

    fig_b, bx = plt.subplots()
    bx.title.set_text('Horizontal Region of Interest (Above Noise)')
    bx.imshow(image[:, index_of_1st_pixel_above_noise_limit_h:index_of_last_pixel_above_noise_limit_flo_h])
    fig_b.savefig("Region_of_Interest_Horizontal.png")

    fig_c, cx = plt.subplots()
    cx.title.set_text('Horizontal & Vertical Region of Interest (Above Noise)')
    cx.imshow(image[index_of_1st_pixel_above_noise_limit_v:index_of_last_pixel_above_noise_limit_flo_v, index_of_1st_pixel_above_noise_limit_h:index_of_last_pixel_above_noise_limit_flo_h])
    fig_c.savefig("Region_of_Interest_Horizontal_Vertical.png")
    plt.close('all')
    return image[index_of_1st_pixel_above_noise_limit_v:index_of_last_pixel_above_noise_limit_flo_v, index_of_1st_pixel_above_noise_limit_h:index_of_last_pixel_above_noise_limit_flo_h]


def plot_horizonal_lineout_intensity(image, coordinates):
    """
    Finds the coordinates (x_max, y_max) of the pixel with the maximum intensity, and plots a line-out of intensity
    values for pixels where y=y_max.

    Args:
        image: The image as a 2D Array
        coordinates: Coordinates you'd like a line-out of.
    Returns:
        (np.ndarray, np.ndarray): A tuple equal to (indices, intensity values). Both components as a np.ndarray
    """
    print("Image Shape: ", image.shape)
    fig_c, cx = plt.subplots()
    cx.title.set_text('Horizontal & Vertical Region of Interest (Above Noise)')
    cx.imshow(image)
    fig_c.savefig("abc.png")
    plt.close('all')
    cv2.imshow("what we are plotting", image)
    x_max, y_max = coordinates
    print("Maxima at: ", (x_max, y_max))
    lineout = np.asarray(image[x_max:x_max+1, :])[0]
    #lineout = np.asarray(image[:, y_max:y_max+1])[0]
    indices = np.arange(0, len(lineout))
    print("Lineout Shape: ", lineout.shape)
    plt.plot(indices, lineout)
    plt.title("Left-Right Trimmed Line-out")
    #plt.show()
    plt.close('all')
    return indices, lineout
def calc_reduced_chi_square(fit, x, y, yerr, N, n_free):
    '''
    fit (array) values for the fit
    x,y,yerr (arrays) data
    N total number of points
    n_free number of parameters we are fitting
    '''
    return 1.0/(N-n_free)*sum(((fit - y)/yerr)**2)


def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))
def chi2(y_measure,y_predict,errors):
    """Calculate the chi squared value given a measurement with errors and prediction"""
    return np.sum( (y_measure - y_predict)**2 / errors**2 )

def chi2reduced(y_measure, y_predict, errors, number_of_parameters):
    """Calculate the reduced chi squared value given a measurement with errors and prediction,
    and knowing the number of parameters in the model."""
    return chi2(y_measure, y_predict, errors)/(y_measure.size - number_of_parameters)


def fit_function(x_coords, y_coords):
    """


    fitter = modeling.fitting.LevMarLSQFitter()
    model = modeling.models.Gaussian1D()   # depending on the data you need to give some initial values
    fitted_model = fitter(model, x_coords, y_coords)
    fig = plt.figure()
    plt.plot(x_coords, y_coords)
    plt.plot(x_coords, fitted_model(x_coords))
    plt.title("Astropy Fit - ./coregistration/cam_b_frame_186.png")
    plt.show()
    fig.savefig('Fit.png')

    plt.close('all')
    print("Astropy Parameters")
    cov_diag = np.diag(fitter.fit_info['param_cov'])
    print('Amplitude: {} +\- {}'.format(fitted_model.amplitude.value, np.sqrt(cov_diag[0])))
    print('Mean: {} +\- {}'.format(fitted_model.mean.value, np.sqrt(cov_diag[1])))
    print('Standard Deviation: {} +\- {}'.format(fitted_model.stddev.value, np.sqrt(cov_diag[2])))
    #scipy stuff

    reduced_chi_squared = calc_reduced_chi_square(fitted_model(x_coords), x_coords, y_coords, np.ones(len(x_coords)), len(x_coords), 3)
    print('Reduced Chi Squared using astropy.modeling: {}'.format(reduced_chi_squared))

        #model_gauss = models.Gaussian1D()
    #fitter_gauss = fitting.LevMarLSQFitter()
    sigma = 1
    y2_err = np.ones(len(x_coords)) * sigma
    popt, pcov = curve_fit(gaussian_distribution, x_coords, y_coords)
    a, b, c = pcov
    best_fit_gauss_2 = np.asarray([gaussian_distribution(x, a, b, c) for x in x_coords])

    print("Scipy Optimize Curve Fit Parameters")
    print('Mean: {} +\- {}'.format(popt[0], np.sqrt(pcov[0, 0])))
    print('Standard Deviation: {} +\- {}'.format(popt[1], np.sqrt(pcov[1, 1])))
    print('Amplitude: {} +\- {}'.format(popt[2], np.sqrt(pcov[2, 2])))
    yEXP = gaussian_distribution(x_coords, *popt)

    reduced_chi_squared = calc_reduced_chi_square(best_fit_gauss_2, x_coords, y_coords, y2_err, len(x_coords), 3)
    print('Reduced Chi Squared using scipy: {}'.format(reduced_chi_squared))

    fig2 = plt.figure()
    plt.plot(x_coords, y_coords, label='Data', marker='o')
    plt.plot(x_coords, yEXP, 'r-', ls='--', label="Exp Fit")
    plt.title("Scipy Optimize Fit - ./coregistration/cam_b_frame_186.png")
    plt.legend()
    plt.show()
    """

    # Create the artificial dataset
    nobs = int(T/dt + 1.5)
    t = dt*np.arange(nobs)
    N = f(t,N0,tau)
    Nfluct = stdev*np.random.normal(size=nobs)
    N = N + Nfluct
    sig = np.zeros(nobs) + stdev

    # Fit the curve
    start = (1100, 90)
    popt, pcov = curve_fit(f,t,N,sigma = sig,p0 = start,absolute_sigma=True)
    print(popt)
    print(pcov)

    # Compute chi square
    Nexp = f(t, *popt)
    r = N - Nexp
    chisq = np.sum((r/stdev)**2)
    df = nobs - 2
    print(“chisq =”,chisq,”df =”,df)

    # Plot the data with error bars along with the fit result
    import matplotlib.pyplot as plt
    plt.errorbar(t, N, yerr=sig, fmt = 'o', label='"data"')
    plt.plot(t, Nexp, label='fit')
    plt.legend()
    plt.show()



img = read_image_from_file("./coregistration/cam_b_frame_186.png")
img_without_noise = crop_surrounding_noise(img, get_coordinates_of_maximum(img))

#plot_horizonal_lineout_intensity(img, get_coordinates_of_maximum(img), upper_limit_noise=10)
xs, ys = plot_horizonal_lineout_intensity(img_without_noise, get_coordinates_of_maximum(img_without_noise))
fit_function(xs, ys)
