import numpy as np
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
from skimage.feature import peak_local_max
'''
load raw data, apply threshold, gaussian smooth, erode, find peaks
'''


plt.switch_backend('agg')  # working on ancient hpc (dlx)


class DiffParser:
    def __init__(self, fig_name='', sgs_name='sgs', threshold=0.15, erode_size=0, region_size=10, peak_flatness=0.8,
                 gaussian_sigma=20, local_max_distance=10):
        """
        :param threshold: threshold for removing noise, default 0.15
        :param erode_size: pixel size for cv eroding, default 10
        :param region_size: peak size must be larger than, default 10
        :param peak_flatness: within peak region the diff between max and min must be smaller than x * max, default 0.8
        :param gaussian_sigma: sigama used in gaussian filter, default 20
        :param local_max_distance: distance between maxima should be larger than, default 10
        """
        self.fig_name = fig_name
        self.sgs_name = sgs_name
        self.qpi, self.qzi, self.image_ls, self.sgs_matrix = self.load_input(self.fig_name, self.sgs_name)
        self.scale_min, self.scale_max = self.color_scaler(self.qpi, self.image_ls)
        # self.image = self.mat2gray(self.image_ls, self.scale_min, self.scale_max)
        self.image = self.mat2gray(self.image_ls, -8, self.scale_max)

        self.threshold = threshold
        self.erode_size = erode_size
        self.region_size = region_size
        self.peak_flatness = peak_flatness
        self.gaussian_sigma = gaussian_sigma
        self.local_max_distance = local_max_distance

    @staticmethod
    def load_input(raw_fig_name, sgs_matrix_name):
        """
        /questions/8172931/
        I don't quite understand Anna's idea about shifting, but I followed her scheme

        :param raw_fig_name: name of matlab *.m figure file
        :param sgs_matrix_name: name of the matlab space group matrix file, default 'sgs' means sgs.mat
        :return:
        """
        matfig = loadmat(raw_fig_name, squeeze_me=True, struct_as_record=False)['hgS_070000']
        image_struct = [i for i in matfig.children.children if i.type == 'image'][0]

        image_a = image_struct.properties.CData  # 2d array np.float64, cannot deal rgb
        in_x = image_struct.properties.XData
        in_y = image_struct.properties.YData
        in_mat = loadmat(sgs_matrix_name)['sgs']

        min1 = abs(np.amin(image_a))
        image_no0 = image_a + min1
        min2 = np.min(image_no0[np.nonzero(image_no0)])
        image_shift = image_no0 + min([min2, abs(min2 - min1)])  # no idea, why not use 1e-12?
        in_logz = np.log(image_shift)
        return in_x, in_y, in_logz, in_mat

    @staticmethod
    def color_scaler(x, z):
        """
        get upper and lower bounds of color scales
        I don't like the breaks but I followed Anna's code

        :param x: x coord
        :param z: 2d array, color in gray scale
        :return: float, float, the upper and lower limits for color scales
        """
        xc = np.argmin(np.abs(x))
        nelem, binedges = np.histogram(z, bins=10)  # auto flat
        bincenters = 0.5 * (binedges[1:] + binedges[:-1])
        nelem_xc, dummy_binedges = np.histogram(z[:, xc], binedges)
        bindx = int(np.argmax(nelem_xc))

        s_llim = np.floor(np.amin(bincenters))
        s_ulim = np.ceil(np.amax(bincenters))

        if bindx > 0:
            for b in range(bindx - 1, 0, -1):
                if nelem_xc[b] < 10:
                    s_llim = np.floor(bincenters[b])
                    break
        if bindx < 9:
            for b in range(9, bindx, -1):
                if nelem_xc[b] > 0:
                    s_ulim = np.ceil(bincenters[b])
                    break
        return s_llim, s_ulim

    def detect_peaks(self):
        """
        using the parameters defined in parser, process the image then find peaks

        :return: smooth_erode: image after processing
                 sorted_peaks: peaks found sorted by distance to origin
        """
        qpi = self.qpi
        qzi = self.qzi
        image = self.image
        th = self.threshold
        erode_size = self.erode_size
        region_size = self.region_size
        peak_flatness = self.peak_flatness
        gaussian_sigma = self.gaussian_sigma
        local_max_distance = self.local_max_distance

        image_thresh = np.copy(image)
        image_thresh[image_thresh < th] = 0
        smooth = filters.gaussian_filter(image_thresh, gaussian_sigma, mode='nearest')
        smooth[smooth < th] = 0
        smooth_erode = cv2.erode(smooth, np.ones((erode_size, erode_size)))
    
        coordinates = peak_local_max(smooth_erode, min_distance=local_max_distance)
        large_peaks = []
        for peak in coordinates:
            xi = peak[0]
            yi = peak[1]
            region = smooth_erode[xi - region_size: xi + region_size, yi - region_size: yi + region_size]
            if np.amax(region) - np.amin(region) < peak_flatness * np.amax(region):
                if yi > 0 and xi > 0:  # large_peaks.append(peak)
                    large_peaks.append([xi, yi])
        large_peaks = np.array(large_peaks)
        large_peaks_coord = np.dstack((qpi[large_peaks[:, 1]], qzi[large_peaks[:, 0]]))[0]
        sorted_index = np.argsort(np.linalg.norm(large_peaks_coord, axis=1))
        sorted_peaks = large_peaks_coord[sorted_index][1:]
        return smooth_erode, sorted_peaks

    @staticmethod
    def mat2gray(mat, smn, smx):
        """
        matlab equivalent normalization

        :param mat:
        :param smn:
        :param smx:
        :return:
        """
        out = (mat - smn) / (smx - smn)
        zeromat = np.zeros(np.shape(out))
        zeromat.fill(0)
        out[out < zeromat] = 0
        zeromat.fill(1)
        out[out > zeromat] = 1
        return out

    def plt_foundpeaks(self):
        """
        plot the detected peaks

        """
        qpi = self.qpi
        qzi = self.qzi
        original = self.image
        smooth_erode, peaks = self.detect_peaks()

        x0 = min(qpi)
        x1 = max(qpi)
        y0 = min(qzi)
        y1 = max(qzi)
        
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        ax1.set_title('original')
        ax1.imshow(original, cmap='jet', origin='lower', extent=[x0, x1, y0, y1])
        
        ax2.set_title('smooth & erode')
        ax2.imshow(smooth_erode, cmap='jet', origin='lower', extent=[x0, x1, y0, y1])
        
        ax3.set_title('peaks found')
        ax3.plot(peaks[:, 0], peaks[:, 1], 'r+')
        ax3.imshow(original, cmap='jet', origin='lower', extent=[x0, x1, y0, y1])

        plt.tight_layout()
        plt.savefig('peaksfound.eps')

# def rgb2gray(rgb):
#     """
#     not used now
#     :param rgb: (n,3) array
#     :return: (n,1) array of gray scale
#     """
#     r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     return gray
