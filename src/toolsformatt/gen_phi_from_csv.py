
import pandas
import numpy as np
from PIL import Image

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)


def gen_phi_imgs(phi_fp, phi_no_bg_fp, phi_background_fp):
    for filename_phi_sample in [phi_fp, phi_no_bg_fp, phi_background_fp]:
        filename_sh_phi_sample = filename_phi_sample.split("/")[-1][:-4]

        phi_sample_csv_file = pandas.read_csv(filename_phi_sample, header=None)
        values_phi_sample = phi_sample_csv_file.values

        where_are_NaNs = np.isnan(values_phi_sample)
        values_phi_sample[where_are_NaNs] = float(0)

        values_sin_phi = np.sin(values_phi_sample)

        SIN_PHI_MATRIX = values_sin_phi


        DISPLAYABLE_PHI_MATRIX = np.zeros((SIN_PHI_MATRIX.shape[0], SIN_PHI_MATRIX.shape[1], 3), dtype=np.uint8)
        DISPLAYABLE_PHI_MATRIX[:, :, 1] = np.where(SIN_PHI_MATRIX < 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)), 0)
        DISPLAYABLE_PHI_MATRIX[:, :, 0] = np.where(SIN_PHI_MATRIX < 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)), 0)

        DISPLAYABLE_PHI_MATRIX[:, :, 0] = np.where(SIN_PHI_MATRIX > 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)),
                                                 DISPLAYABLE_PHI_MATRIX[:, :, 0])


        image = Image.fromarray(DISPLAYABLE_PHI_MATRIX.astype('uint8'), 'RGB')
        image.save(filename_phi_sample.replace(".csv", ".png"))

def gen_phi_imgs2(phi_fp, phi_background_fp):
    for filename_phi_sample in [phi_fp, phi_background_fp]:
        filename_sh_phi_sample = filename_phi_sample.split("/")[-1][:-4]

        phi_sample_csv_file = pandas.read_csv(filename_phi_sample, header=None)
        values_phi_sample = phi_sample_csv_file.values

        where_are_NaNs = np.isnan(values_phi_sample)
        values_phi_sample[where_are_NaNs] = float(0)

        values_sin_phi = np.sin(values_phi_sample)

        SIN_PHI_MATRIX = values_sin_phi


        DISPLAYABLE_PHI_MATRIX = np.zeros((SIN_PHI_MATRIX.shape[0], SIN_PHI_MATRIX.shape[1], 3), dtype=np.uint8)
        DISPLAYABLE_PHI_MATRIX[:, :, 1] = np.where(SIN_PHI_MATRIX < 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)), 0)
        DISPLAYABLE_PHI_MATRIX[:, :, 0] = np.where(SIN_PHI_MATRIX < 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)), 0)

        DISPLAYABLE_PHI_MATRIX[:, :, 0] = np.where(SIN_PHI_MATRIX > 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)),
                                                 DISPLAYABLE_PHI_MATRIX[:, :, 0])


        image = Image.fromarray(DISPLAYABLE_PHI_MATRIX.astype('uint8'), 'RGB')
        image.save(filename_phi_sample.replace(".csv", ".png"))
