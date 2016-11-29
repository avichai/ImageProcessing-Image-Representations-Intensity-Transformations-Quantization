import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
import scipy.stats as stats
from skimage.color import rgb2gray
import os

# ==============================================================================
# used for the bonus only!
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
# ==============================================================================

GRAY = 1
RGB = 2
NORM_PIX_FACTOR = 255
DIM_RGB = 3
MAX_PIX_VALUE = 256
MIN_PIX_VALUE = 0
Y = 0
ROWS = 0
COLS = 1
FUNC_HIST = 1
FUNC_QUANT = 2

YIQ_TRANS_ARRAY = np.array(
    [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]) \
    .astype(np.float32)
RGB_TRANS_ARRAY = np.linalg.inv(YIQ_TRANS_ARRAY).astype(np.float32)


def read_image(filename, representation):
    """this function reads a given image file and converts it into a given 
    representation:
    filename - string containing the image filename to read.
    representation - representation code, either 1 or 2 defining if the 
                     output should be either a grayscale image (1) or an 
                     RGB image (2).
    output - the image in the given representation when the pixels are
             of type np.float32 and normalized"""
    filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        return
    im = imread(filename)
    if im.dtype == np.float32:
        '''I don't handle this case, we asume imput in uint8 format'''
        return
    if representation == GRAY:
        im = rgb2gray(im).astype(np.float32)
        if (im.max() > 1):
            im = im.astype(np.float32)
            im /= NORM_PIX_FACTOR
        return im
    im = im.astype(np.float32)
    im /= NORM_PIX_FACTOR
    return im


def imdisplay(filename, representation):
    """ This function display a given image file and in the given representation
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining if the
                     output should be either a grayscale image (1) or an
                     RGB image (2)."""
    im = read_image(filename, representation)
    plt.figure()
    if representation == GRAY:
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)
    plt.show()


def rgb2yiq(imRGB):
    """convert a RGB image to its YIQ image
    :param imRGB: image(with pixels in range [0,1] that are np.float32) """
    if len(imRGB.shape) != DIM_RGB:
        return imRGB
    dim_im, numPix, dimX, dimY = getImParams(imRGB)
    imRGB_flatten = imRGB.reshape(numPix, DIM_RGB).transpose()
    return np.dot(YIQ_TRANS_ARRAY, imRGB_flatten).transpose() \
        .reshape(dimX, dimY, DIM_RGB)


def yiq2rgb(imYIQ):
    """convert a YIQ image to its RGB image
    :param imYIQ: image (with pixels in range [0,1] that are np.float32) """
    if len(imYIQ.shape) != DIM_RGB:
        return imYIQ
    dim_im, numPix, dimX, dimY = getImParams(imYIQ)
    imYIQ_flatten = imYIQ.reshape(numPix, DIM_RGB).transpose()
    return np.dot(RGB_TRANS_ARRAY, imYIQ_flatten).transpose() \
        .reshape(dimX, dimY, DIM_RGB)


def getImParams(im_orig):
    """gets the dimensions of the image and the number of pixels"""
    im_shape = im_orig.shape
    dim_im = len(im_shape)
    dimX = im_shape[ROWS]
    dimY = im_shape[COLS]
    numPix = dimX * dimY
    return dim_im, numPix, dimX, dimY


def getWorkIm(im_orig, dim_im):
    """return an image in the right range of the pixels (np.float 32
    that are normalized)"""
    if dim_im == DIM_RGB:
        yiq_im = rgb2yiq(im_orig)
        im = yiq_im[:, :, Y]
    else:
        im = im_orig
    im_ints = np.around((im * NORM_PIX_FACTOR)).astype(np.uint8)
    return im_ints


def getHistAndCumsum(im_ints):
    """get the histogram and cumsum of an image that is in its ints 
    representation (np.uint8)"""
    hist_orig, bin_edges_orig = np.histogram \
        (im_ints, bins=MAX_PIX_VALUE, range=[MIN_PIX_VALUE, MAX_PIX_VALUE])
    cumsum = np.cumsum(hist_orig)
    return hist_orig, cumsum


def strechNormCumsum(normCumsum, newMin, newMax):
    """lineary strech the normCumsum into its new borders"""
    inMin = np.min(normCumsum)
    inMax = np.max(normCumsum)
    normCumsum = (normCumsum - inMin) * \
                 ((newMax - newMin) / (inMax - inMin)) + newMin
    return normCumsum


def getOutputIm(im_eq_ints, im_orig, dim_im, func):
    """return the image after converting it to the right type and normalize
    (np.float32) and back to rgb if need to. expect an image in np.uint8
    pixels, the original image that im_eq_ints was derived from and the 
    dimensions"""
    im_eq = im_eq_ints.astype(np.float32)
    im_eq /= NORM_PIX_FACTOR
    # im_eq = np.clip(im_eq, 0, 1)
    if dim_im == DIM_RGB:
        yiq_im = rgb2yiq(im_orig)
        yiq_im[:, :, Y] = im_eq
        im_eq = yiq2rgb(yiq_im)
    if func == FUNC_HIST:
    	return np.clip(im_eq, 0, 1)
    else:
    	return im_eq


def histogram_equalize(im_orig):
    """performs histogram equalization of a given grayscale or RGB image
    :param im_orig: is the input grayscale or RGB float32 image with values in
    [0, 1].
    output:
    im_eq - is the equalized image. grayscale or RGB float32 image with values 
    in [0, 1].
    hist_orig - is a 256 bin histogram of the original image (array with shape 
    (256,) ).
    hist_eq - is a 256 bin histogram of the equalized image (array with shape 
    (256,) )."""
    dim_im, numPix, dimX, dimY = getImParams(im_orig)
    im_ints = getWorkIm(im_orig, dim_im)
    hist_orig, cumsum = getHistAndCumsum(im_ints)
    normCumsum = (cumsum / numPix * (MAX_PIX_VALUE - 1))
    if (np.min(normCumsum) != MIN_PIX_VALUE) or \
            (np.max(normCumsum) != MAX_PIX_VALUE - 1):
        normCumsum = strechNormCumsum(normCumsum, MIN_PIX_VALUE,
                                      (MAX_PIX_VALUE - 1))
    normCumsum = np.around(normCumsum).astype(np.uint8)
    im_eq_ints = normCumsum[im_ints].astype(np.uint8)
    hist_eq, bin_edges_eq = np.histogram(im_eq_ints, bins=MAX_PIX_VALUE,
                                         range=[MIN_PIX_VALUE, MAX_PIX_VALUE])
    im_eq = getOutputIm(im_eq_ints, im_orig, dim_im, FUNC_HIST)
    return [im_eq, hist_orig, hist_eq]


def fitZandQ(n_iter, n_quant, z, q, error, hist_orig, g):
    """fits the z and q to the data using less or equal to n_iter iteration
    and outputting the final z, q and the error vector (composed of the
    error in each iteration)"""

    """the biwize multiplication of the histogram and the g value [0,255]"""
    hMulG = hist_orig * g
    z_old = z.copy()
    for j in range(n_iter):
        curError = 0
        for i in range(n_quant):
            """corresponding to the formula learned in class"""
            sumOfHmulG = np.sum(hMulG[z[i]:z[i + 1]])
            sumG = np.sum(hist_orig[z[i]:z[i + 1]])
            if sumG != 0:
                q[i] = sumOfHmulG / sumG
            else:
                '''so the program will not crash'''
                q[i] = (z[i] + z[i + 1]) / 2
        for i in range(n_quant - 1):
            z[i + 1] = np.ceil((q[i] + q[i + 1]) / 2)
        for i in range(n_quant):
            curError += np.sum(
                ((q[i] - g)[z[i]:z[i + 1]] ** 2) * (hist_orig[z[i]:z[i + 1]]))
        error.append(curError)
        if (z == z_old).all():
            """we've converged"""
            break
        else:
            z_old = z.copy()
    z[len(z) - 1] = MAX_PIX_VALUE - 1
    q = np.around(q)
    return z, q, error


def quantize(im_orig, n_quant, n_iter):
    """performs optimal quantization of a given grayscale or RGB image
    :param im_orig: is the input grayscale or RGB image to be quantized (float32
    image with values in [0, 1]).
    :param n_quant: is the number of intensities your output im_quant image
    should have.
    :param n_iter: is the maximum number of iterations of the optimization
    procedure (may converge earlier.)
    output:
    im_quant - is the quantize output image.
    error - is an array with shape (n_iter,) (or less) of the total 
    intensities error for each iteration in the
    quantization procedure."""
    if n_quant <= 0:
        return
    dim_im, numPix, dimX, dimY = getImParams(im_orig)
    im_ints = getWorkIm(im_orig, dim_im)

    hist_orig, cumsum = getHistAndCumsum(im_ints)

    '''fixing initial z'''
    z = stats.mstats.mquantiles(im_ints, prob=np.linspace(0, 1, n_quant + 1)) \
        .astype(np.uint16)
    z[0] = 0
    z[-1] = MAX_PIX_VALUE
    q = np.array(np.zeros(n_quant), dtype=np.float32)
    error = []
    g = np.arange(MAX_PIX_VALUE)
    """fitting the z and q to the data"""
    z, q, error = fitZandQ(n_iter, n_quant, z, q, error, hist_orig, g)
    im_quant_t = np.zeros(im_ints.shape).astype(np.uint8)
    for i in range(n_quant):
        """creating the map map of the quantization"""
        im_quant_t[im_ints <= z[n_quant - i]] = n_quant - 1 - i
    im_quant_ints = q[im_quant_t].astype(np.uint8)
    im_quant = getOutputIm(im_quant_ints, im_orig, dim_im, FUNC_QUANT)
    return [im_quant, np.array(error, dtype=np.float32)]


def recreate_image(cluster_centers, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = cluster_centers.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = cluster_centers[labels[label_idx]]
            label_idx += 1
    return image.astype(np.float32)


def quantize_rgb(im_orig, n_quant, n_iter):
    """Bonus:
    quantize_rgb using kmeans algorithm 
    im_orig - is the input grayscale or RGB image to be quantized (float32 
    image with values in [0, 1]).
    n_quant - is the number of intensities your output im_quant image 
    should have.
    n_iter - is the maximum number of iterations of the optimization procedure 
    (may converge earlier.)
    output:
    im_quant - is the quantize output image.
    error - is an array with shape (n_iter,) (or less) of the total
    intensities error for each iteration in the
    quantization procedure."""
    n_colors = n_quant
    im = np.array(im_orig)
    w, h, d = tuple(im.shape)
    assert d == DIM_RGB
    """flatten the image according to w and h"""
    image_array = np.reshape(im, (w * h, d))
    """taking a sample of the pixels (assume that the image is more than
    32*32)"""
    image_array_sample = shuffle(image_array, random_state=0)[:1024]
    error = []
    oldError = 0.0
    for i in range(n_iter):
        kmeans = KMeans(n_clusters=n_colors, max_iter=(i + 1),
                        random_state=0).fit(image_array_sample)
        """getting the error"""
        newError = kmeans.inertia_
        if newError == oldError:
            '''we've converged'''
            break
        error = error + [newError]
    """predicting the full image"""
    labels = kmeans.predict(image_array)
    return [recreate_image(kmeans.cluster_centers_, labels, w, h),
            np.array(error, dtype=np.float32)]
