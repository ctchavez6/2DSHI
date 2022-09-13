"""
REFERENCED ARTICLES
https://towardsdatascience.com/image-processing-with-python-application-of-fourier-transformation-5a8584dc175b
https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html#and-n-d-discrete-fourier-transforms
https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_fft_image_denoise.html
https://www.astrobetter.com/wiki/tiki-index.php?page=python_radial_profiles for the "radial profile function"
"""
import numpy as np
import pandas as pd
from scipy import fftpack, ndimage
from matplotlib.colors import LinearSegmentedColormap
from skimage.color import rgb2gray
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
import radialProfile
# assign names, read csv data files
phi_csv = 'phi_bg_avg_5.csv'
phi_data = pd.read_csv(phi_csv, header=None)
data = phi_data.values
phi_gray = rgb2gray(data)
data_less_mean = data - np.mean(data) # SUBTRACT THE AVERAGE NOISE

FFT_shifted = np.fft.fftshift(np.fft.fft2(data)) #shift the FFT's zero to the image center
FFT_unshifted = np.fft.fft2(data)
reconstructed_Sample = fftpack.ifft2(FFT_unshifted).real # reconstruct the
# FFT_unshifted_less_mean = np.fft.fft2(data_less_mean)
# spectrum_wo_DC = np.fft.fftshift(np.fft.fft2(data_wo_DC))

r, c = FFT_shifted.shape # get the shape of the FFT image
keep_fraction = .46 # how much of the 1/2 image to delete
FFT2_shifted = np.fft.fftshift(np.fft.fft2(data)) # create a duplicate of (spectrum) FFT
FFT2_shifted[:, 0:int(r * keep_fraction)] = 0 # zero out selected columns
FFT2_shifted[:, int(r * (1 - keep_fraction)):] = 0
FFT2_shifted[0:int(r * keep_fraction), :] = 0  # zero out selected rows
FFT2_shifted[int(r * (1 - keep_fraction)):, :] = 0
FFT2_unshifted = np.fft.ifftshift(FFT2_shifted) # subtract the mean noise level
FFT2_shifted_less_mean = np.fft.ifftshift(FFT2_shifted - np.mean(data))
reconstructed_masked_Sample = fftpack.ifft2(FFT2_unshifted).real
# reconstructed_masked_Sample_less_mean = fftpack.ifft2(FFT2_shifted_less_mean).real

keep_fraction2 = 0.10 # how much of the 1/2 image to delete
FFT3_shifted = np.fft.fftshift(np.fft.fft2(data)) # create a duplicate of (spectrum) FFT
FFT3_shifted[:, 0:int(r * keep_fraction2)] = 0 # zero out selected columns
FFT3_shifted[:, int(r * (1 - keep_fraction2)):] = 0
FFT3_shifted[0:int(r * keep_fraction2), :] = 0  # zero out selected rows
FFT3_shifted[int(r * (1 - keep_fraction2)):, :] = 0
FFT3_unshifted = np.fft.ifftshift(FFT3_shifted) # subtract the mean noise level
FFT3_shifted_less_mean = np.fft.ifftshift(FFT3_shifted - np.mean(data))
reconstructed_masked_Sample2 = fftpack.ifft2(FFT3_unshifted).real
# reconstructed_masked_Sample_less_mean = fftpack.ifft2(FFT2_shifted_less_mean).real
# HERE

freqx=np.fft.fftshift(np.fft.fftfreq(1001,1))
freqy=np.fft.fftshift(np.fft.fftfreq(1001,1))
fX,fY= np.meshgrid(freqx,freqy)

# normal average of the original image Sample
avg_size = 30
Sample_image_averaged = ndimage.uniform_filter(data, size = avg_size)

# Calculate the 2D power spectrum
psd2D = np.abs(FFT_shifted) ** 2
psd2D_1 = np.abs(FFT2_shifted) ** 2
psd2D_3 = np.abs(FFT3_shifted) ** 2
FFT_Sample_image_averaged = np.fft.fftshift(np.fft.fft2(Sample_image_averaged))

psd2D_2 = np.abs(FFT_Sample_image_averaged) ** 2 #don't know what this is

# Calculate the azimuthally averaged 1D power spectrum
psd1D = radialProfile.azimuthalAverage(psd2D)
psd1D_1 = radialProfile.azimuthalAverage(psd2D_1)
psd1D_2 = radialProfile.azimuthalAverage(psd2D_2)
psd1D_3 = radialProfile.azimuthalAverage(psd2D_3)

x, y = data.shape
X,Y= np.meshgrid(x,y)

colors = [(1,0,0), (0,0,0), (1,1,0)]
n_bins = 100
cmap_name = 'shdi' #define a color map name
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

plt.figure(figsize=(25,15))
ax1 = plt.subplot(331)
im1 = plt.imshow(data, cmap=cmap)
plt.title('a) Original Image')
# # create right side sub-ax. width of cax is 5% of ax, padding is fixed at 0.05 inch.
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax)

ax2 = plt.subplot(332)
im2 = plt.imshow(reconstructed_masked_Sample2, cmap=cmap)
plt.title('b) FFT {:.0f}% Masked'.format((2 * keep_fraction2) * 100))
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax)

ax3 = plt.subplot(333)
im3 = plt.imshow(reconstructed_masked_Sample, cmap=cmap)
plt.title('c) FFT {:.0f}% Masked'.format((2 * keep_fraction) * 100))
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax)

ax4 = plt.subplot(334)
# im2 = plt.pcolormesh(fX,fY,np.abs(spectrum), shading='auto')
im4 = plt.imshow(np.log(abs(FFT_shifted)), cmap='gray')
plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=5)
z = ['750', '-500', '-250', '0', '250', '500']
ax4.set_xticklabels(z)
ax4.set_yticklabels(z)
plt.title('d) Orig. Image FFT, <bkg> = {:.4f}'.format(np.mean(data)))
divider = make_axes_locatable(ax4)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im4, cax=cax)

ax5 = plt.subplot(335)
im5 = plt.imshow(np.log(abs(FFT3_shifted)), cmap='gray')
plt.title('e) FFT {:.0f}% Masked'.format((2 * keep_fraction2) * 100))
# plt.title('FFT for the Masked Sample')
plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=5)
z = ['750', '-500', '-250', '0', '250', '500']
ax5.set_xticklabels(z)
ax5.set_yticklabels(z)
divider = make_axes_locatable(ax5)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im5, cax=cax)

ax6 = plt.subplot(336)
im6 = plt.imshow(np.log(abs(FFT2_shifted)), cmap='gray')
plt.title('f) FFT {:.0f}% Masked'.format((2 * keep_fraction) * 100))
# plt.title('FFT for the Masked Sample')
plt.locator_params(axis="x", nbins=5)
plt.locator_params(axis="y", nbins=5)
z = ['750', '-500', '-250', '0', '250', '500']
ax6.set_xticklabels(z)
ax6.set_yticklabels(z)
divider = make_axes_locatable(ax6)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im6, cax=cax)

ax7 = plt.subplot(337)
im7 = plt.imshow(Sample_image_averaged, cmap=cmap)
plt.title('g) Averaged Sample, size = {:.0f}'.format(avg_size))
divider = make_axes_locatable(ax7)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im7, cax=cax)

ax8 = plt.subplot(338)
im8 = plt.semilogy(psd1D, label='Original Sample')
im8 = plt.semilogy(psd1D_3, label='{:.0f}% Masked FFT'.format((2*keep_fraction2)*100))
im8 = plt.semilogy(psd1D_1, label='{:.0f}% Masked FFT'.format((2*keep_fraction)*100))
im8 = plt.semilogy(psd1D_2, label='{:.0f} Average Sample'.format(avg_size))
legend = ax8.legend(loc='upper right', fontsize='large')
plt.title('h) Power Spectra')

plt.suptitle('Image Filtration, Comparing Masked FFTs and Sample Averaging', x= 0.5, y = 0.95, fontsize=16)
# plt.tight_layout()
plt.show()


# write image data to csv files
df1 = pd.DataFrame(reconstructed_masked_Sample)
df2 = pd.DataFrame(Sample_image_averaged)
df1.to_csv('_ReconFromMaskedFFT.csv', index=None, header=True)
df2.to_csv('_AveragedImage.csv', index=None, header=True)

# # Plotly plots, uncomment if desired
# sh_1, sh_2 = phi_data.shape
# x1, y1 = np.linspace(0, sh_1, sh_1), np.linspace(0, sh_2, sh_2)
# x2, y2 = np.linspace(0, sh_1, sh_1), np.linspace(0, sh_2, sh_2)
#
# fig = make_subplots(rows=2, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}], [{"type": "surface"}, {"type": "surface"}]], subplot_titles=('a','b','c','d'))
# fig.add_trace(go.Surface(z=phi_data, x=x1, y=y1, colorscale='Plotly3', colorbar=dict(title='phi', len=0.5, x=0.4, y = 0.75)),row=1, col=1)
# fig.add_trace(go.Surface(z=np.log(FFT_shifted.real), x=x2, y=y2, colorscale='Plotly3', colorbar=dict(title='log(fft amp)', len=0.5, x=1.0, y = 0.75)),row=1, col=2)
# fig.add_trace(go.Surface(z=np.log(FFT2_shifted.real), x=x2, y=y2, colorscale='Plotly3', colorbar=dict(title='log(fft amp)', len=0.5, x=1.0, y = 0.75)),row=2, col=1)
# fig.add_trace(go.Surface(z=reconstructed_masked_Sample, x=x2, y=y2, colorscale='Reds', colorbar=dict(title='phi', len=0.5, x=1.0, y = 0.25)),row=2, col=2)
# fig.update_layout(title=("a. Sample, b. Sample FFT, c. Masked FFT, d. Recon.Masked Sample"), autosize=True,
#                   width=1500, height=800, margin=dict(l=20, r=20, b=50, t=90))
# fig.show()