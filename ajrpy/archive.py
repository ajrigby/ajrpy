#
# Date:     6th September 2016
#
# Author:   Andrew Rigby
#
# Purpose:  A repository for useful stuff
#
# Version:  v1

"""
----------------------- List of functions: ------------------------------------
ajrectangle - rotated rectangular patch
AzimuthalAverage -
BB93 - Brand & Blitz (1993) rotation curve
BB93a - Brand & Blitz (1993) rotation curve
beamarea -
beameff30m - returns beam efficiences for the IRAM 30-m telescope
CentredDistanceMatrix -
confidence_ellipse -
confidence_ellipse2 -
cursorindex - call after a figure to see data indices upon mouseover
extract_N2_extension - extract a single NIKA2 map from an IDL-made fits file.
friends_of_friends - run a friends of friends algorithm to cluster data
galxy - return x and y coordinates of l and b coord given a wcs object
get_aspect - returns the aspect ratio of an axis element
GreyBody - return intensity for a greybody for given freq and temp
highpassimage - returns a high-pass filtered image
image - produces a simple subplot for displaying an image
index2velocity - return the velocity of a channel given wcs
Jy2K - converts an astropy quantity from Jy-like units to K
K2Jy - converts an astropy quantity from units of K to Jy
kd_URC - Universal rotation curve used in the kd function
kd - return near and far kinematic distances based on Reid+19 and the ^^ URC ^^
linfit -
linfunc -
mad - median absolute deviation
makeheader - make a simple WCS header
mcodr - Monte Carlo orthogonal distance regression
mJybeam2MJysr - unit conversion based on beam size
MJysr2mJybeam - unit conversion based on beam size
normal - return a normal distribution for a given array of 'x' values
nrtau - Calculate an LTE optical depth from two lines at given abundance ratio
partial_corr - Calculate a partial correlation coefficient
pointflux - gives the flux of a point source with given mass at given freq
planck - returns the value of the Planck(nu, T) function in MJy/sr
Planck - returns the value of a Planck(nu, T) function as an astropy quantity
pointmass - gives teh mass of a point source with given flux at given freq
powerspace - generate levels for e.g. contours with a power law index
PowerSpectrum - calculate a 1D power spectrum of an image
radex_input - prepare input files for radex
radex_output - read output of radex file
radex - run radex and return T_ex, tau, T_rad, integrated intensity
rchisq -
resolvedmass -
rms - calculates the root mean squared value of an array
round_up_to_odd
smoothcube - smooth a cube
smoothimage - smooth an image
Tcol - returns the colour temperature for fluxes at e.g. 160 & 250 microns
truncate_colormap
velocity2index - gives the velocity of a index with given wcs
wcscutout - cuts out a region of a map and produces adjusted wcs object
X12C13CR - Carbon isotope ratio as a function of Galactocentric distance

----------------------- List of classes: --------------------------------------
MonteCarlo - for Monte Carlo error propagation for an input function
"""

import datetime
import numpy as np
import numpy.random as rnd
import numpy.fft as fft

import scipy.constants as sc
import scipy.stats as stat
import scipy.odr.odrpack as odrpack
from scipy import stats, linalg, interpolate
import scipy.optimize as optimization

from astropy.convolution import Gaussian2DKernel, convolve_fft, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling.models import BlackBody
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy import units as u
from astropy import constants as ac

import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes
import mplcursors

from sklearn.cluster import AgglomerativeClustering

import cmocean.cm as co
import os

h = sc.h
c = sc.c
Msol = 2.E30	     # kg
sig2fwhm = np.sqrt(8. * np.log(2))
fwhm2sig = 1 / sig2fwhm


# ========== Functions in alphabetical order ================================ #
def ajrectangle(CenX, CenY, Xsize, Ysize, PA, fc='none', ec='k', **kwargs):
    """
    Purpose:
        Wrapper for a matplotlib.patches.Rectangle instance, which
        interprets the PA as a rotation about the rectangle centre.
    Arguments:
        CenX, CenY - Central coordinates
        Xsize, Ysize - Sizes
        PA - Position angle (degrees east of north)
        **kwargs - keyword arguments for matplotlib.patches object
    Returns:
        matplotlib.patches object
    """
    from matplotlib.patches import Rectangle
    Xcoord = -0.5 * Xsize
    Ycoord = -0.5 * Ysize
    BLx = CenX + (Xcoord * np.cos(np.deg2rad(PA))) -\
        (Ycoord * np.sin(np.deg2rad(PA)))
    BLy = CenY + (Xcoord * np.sin(np.deg2rad(PA))) +\
        (Ycoord * np.cos(np.deg2rad(PA)))
    Patch = Rectangle((BLx, BLy), Xsize, Ysize, angle=PA, fc=fc, ec=ec,
                      **kwargs)
    return Patch


def AzimuthalAverage(image, center=None):
    """
    Purpose:
        Calculate the azimuthally averaged radial profile.
    Arguments:
        image - The 2D image
        center - The [x,y] pixel coordinates used as the center. The default is
            None, which then uses the center of the image (including fracitonal
            pixels).
    Returns:
    Notes:
        Adapted from Jessica Lu's astrobetter blog:
        http://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max() - x.min()) / 2, (x.max() - x.min()) / 2])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def BB93(lon, lat, vel):
    """
    Purpose:
        Calculate near and far kinematic distances from Brand & Blitz (1993)
    Arguments:
        Beam FWHM
    Returns:
        Area, in units of units(FWHM)**2. Input as e.g. arcsec or pixels
    Notes:
    """
    astropyunits = True
    if not hasattr(lon, 'unit'):
        astropyunits = False
        lon *= u.deg
    if not hasattr(lat, 'unit'):
        lat *= u.deg
    if not hasattr(vel, 'unit'):
        vel *= u.km / u.s

    a1 = 1.0077
    a2 = 0.0394
    a3 = 0.0071
    theta0 = 220 * u.km / u.s
    R0 = 8.5 * u.kpc
    omega0 = theta0 / R0

    if vel.value <= 20:
        Rint = 6.5 * u.kpc
    if (vel.value > 20) & (vel.value < 60):
        Rint = 4.5 * u.kpc
    if vel.value >= 60:
        Rint = 3.5 * u.kpc

    R = Rint
    vdiff = 10 * u.km / u.s

    while np.abs(vdiff) > 0.05 * u.km / u.s:
        R += 0.001 * u.kpc
        omega = omega0 * ((a1 * (R / R0)**(a2 - 1)) + (a3 * (R0 / R)))
        vcalc = R0 * (omega - omega0) * np.sin(lon) * np.cos(lat)
        vdiff = vcalc - vel

    # Distance "Quadratic Co-efficients"
    aq = (np.cos(lat))**2
    bq = 2 * R0 * np.cos(lat) * np.cos(lon)
    cq = (R0**2) - (R**2)

    if ((bq**2) - 4 * aq * cq) < 0:
        d1 = (R0 * np.cos(lon)) / np.cos(bat)
        d2 = (R0 * np.cos(lon)) / np.cos(bat)
    else:
        d1 = -((-bq + np.sqrt((bq**2) - (4 * aq * cq))) / (2 * aq))
        d2 = -((-bq - np.sqrt((bq**2) - (4 * aq * cq))) / (2 * aq))

    if not astropyunits:
        return d1.value, d2.value
    else:
        return d1, d2


def BB93a(lon, lat, vel, R0=8.5 * u.kpc, Theta0=220 * u.km / u.s):
    """
    Purpose:
        Calculate near and far kinematic distances from Brand & Blitz (1993)
    Arguments:
        Beam FWHM
    Returns:
        Area, in units of units(FWHM)**2. Input as e.g. arcsec or pixels
    Notes:
    """
    if not hasattr(lon, 'unit'):
        lon *= u.deg
    if not hasattr(lat, 'unit'):
        lat *= u.deg
    if not hasattr(vel, 'unit'):
        vel *= u.km / u.s

    a1 = 1.0077
    a2 = 0.0394
    a3 = 0.0071
    omega0 = Theta0 / R0        # The Sun's angular velocity

    Rgc = np.arange(0, 20, 0.0001) * u.kpc   # Sample to nearest 0.1 pc
    ratio = Rgc / R0
    omega = omega0 * (a1 * (ratio**(a2 - 1)) + a3 * (ratio**(-1)))
    vposs = R0 * (omega - omega0) * np.sin(lon) * np.cos(lat)

    Rbest = Rgc[np.argmin(np.abs(vposs - vel))]
    Rmin = 2 * u.kpc
    Rmax = 17 * u.kpc

    if Rbest < Rmin:
        Rbest = np.nan
    if Rbest > Rmax:
        Rbest = np.nan

    # Quadratic formula
    aq = np.cos(lat)**2
    bq = -2 * R0 * np.cos(lat) * np.cos(lon)
    cq = R0**2 - Rbest**2

    if bq**2 - 4 * aq * cq < 0 * u.kpc**2:
        dnear = R0 * np.cos(lon) / np.cos(lat)
        dfar = R0 * np.cos(lon) / np.cos(lat)
    else:
        dfar = (-bq + np.sqrt(bq**2 - 4 * aq * cq)) / (2 * aq)
        dnear = (-bq - np.sqrt(bq**2 - 4 * aq * cq)) / (2 * aq)
    if not hasattr(lon, 'unit'):
        return dnear.value, dfar.value
    else:
        return dnear, dfar


def beamarea(beamfwhm):
    """
    Purpose:
        Give the area of a Gaussian beam with given fwhm
    Arguments:
        Beam FWHM
    Returns:
        Area, in units of units(FWHM)**2. Input as e.g. arcsec or pixels
    Notes:
    """
    return beamfwhm**2. * np.pi / (4. * np.log(2.))


def beameff30m(freq):
    """
    Return efficiences for the 30-m telesope by interpolating the values on
    https://publicwiki.iram.es/Iram30mEfficiencies using a cubic spline
    Arguments:
        freq - frequency in GHz
    Returns:
        (Beff, Feff)

    Notes: TMB = (Feff / Beff) x TA*
    """
    Freq0 = np.array([86, 115, 145, 210, 230, 280, 340, 345])
    Beff0 = np.array([81, 78, 73, 63, 59, 49, 35, 34]) / 100
    Feff0 = np.array([95, 94, 93, 94, 92, 87, 81, 80]) / 100

    Beff = interpolate.CubicSpline(Freq0, Beff0)(freq)
    Feff = interpolate.CubicSpline(Freq0, Feff0)(freq)
    return Beff, Feff


def CentredDistanceMatrix(sizex, sizey):
    """
    Purpose:
        Returns a matrix with intensity equal to the distance (in pixels) from
        a central point.
    Arguments:
        Array with the desired dimension, x,y coordinates of the central point
    Returns:
        A matrix with intensity equal to the distance (in pixels) from the
        central pixel
    # Notes:
    #     The sizes should be odd
    """
    cenx = (sizex - 1) / 2
    ceny = (sizey - 1) / 2
    y, x = np.meshgrid(range(sizey), range(sizex))
    return np.sqrt((x - cenx)**2 + (y - ceny)**2)





def clean_axes(ax):
    """
    Purpose:
        Remove ticks, tick labels, and axis labels from a 2D imshow axes object
        for a minimalist appearance.
    Argumments:
        ax - the axes object to be cleaned
    """
    if hasattr(ax, 'wcs'):
        ax1, ax2 = ax.coords
        ax1.set_ticks_visible(False)
        ax1.set_ticklabel_visible(False)
        ax2.set_ticks_visible(False)
        ax2.set_ticklabel_visible(False)
        ax1.set_axislabel('')
        ax2.set_axislabel('')
    else:
        for axi in [ax.xaxis, ax.yaxis]:
            axi.set_label('')
            axi.set_ticks([])
            axi.set_ticklabels([])
            # ax.set_tick_params(which='both', visible=False)


def confidence_ellipse(x, y, ax, n_std=3, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    From https://matplotlib.org/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def confidence_ellipse2(x, y, ax, n_std=3, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    From https://matplotlib.org/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ellipse, ell_radius_x, ell_radius_y


def cursorindex():
    """
    Call following a matplotlib plot to give a mouse-over that lists the index
    of the targeted data point.
    """
    mplcursors.cursor().connect(
        "add",
        lambda sel: sel.annotation.set_text(format(sel.target.index, ".0f")))


def extract_N2_extension(fits_file_in, fits_file_out, ext, overwrite=True):
    """
    Purpose:
        Extract an extension from a NIKA2 fits HDU.
    Arguments:
        Input & output fits filenames (as strings), extension to extract.
    Returns:
        Writes the desired extension to a fits file.
    Notes:

    """
    hdu = fits.open(fits_file_in)
    primary_header = hdu[0].header
    Data = hdu[ext].data
    ext_header = hdu[ext].header
    ext_header += primary_header
    wcs = WCS(hdu[1].header)
    if (ext != 1) or (ext != 4):
        ext_header += wcs.to_header()
    hdu_out = fits.PrimaryHDU(Data, ext_header)
    hdu_out.writeto(fits_file_out, overwrite=overwrite)


def friends_of_friends(data, linking_length):
    """
    Performs a simple friends-of-friends matching where data points are joined
    if they fall within some linking length of a neighbour. This produces the
    same results as TOPCAT's Internal Match using the N-d Cartesian Anisotropic
    algorithms.

    Arguments:
        data: a list of coordinates for each axis, e.g. [glon, glat, vlsr]
              The list is of N dimensions and M data points.
        linking_length: list of search tolerances in each axis, in the same
            units as data. E.g. [0.0008, 0008, 0.4]. Must have N dimensions.
            This is equivalent to the "Error in X/Y/Z" function in TOPCAT.
    Returns:
        labels: A list of size M containing the cluster ID for each data point
    """
    data_in = np.array(data) / np.array(linking_length).reshape(-1, 1)

    clusterer = AgglomerativeClustering(n_clusters=None, linkage='single',
                                        distance_threshold=1)
    clusters = clusterer.fit(data_in.T)
    GroupID = clusters.labels_
    GroupSize = np.array([len(np.where(GroupID == x)[0]) for x in GroupID])
    return GroupID, GroupSize


def galxy(lon, lat, wcsobj):
    x, y = wcsobj.all_world2pix(lon, lat, 1)
    return (x, y)


def get_aspect(ax=None):
    """
    Gets the aspect ratio of a pair of axes
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    axes_ratio = height / width
    aspect = axes_ratio / ax.get_data_ratio()

    return aspect


def GreyBody(nu, NH2, Td, beta=1.8, mumol=2.8,
             k0=0.12 * u.cm**2 / u.g, nu0=1200 * u.GHz):
    """
    Date added:
        27 September 2018
    Purpose:
        Returns the specific intensity for a modified black body model. Will
        return an astropy Quantity object if the input nu, NH2 and Td are.
    Arguments:
        nu  - Frequency at which to determine specific intensity [GHz]
        NH2 - Total H2 column density along the line of sight [cm-2]
        Td  - Dust temperature [K]
    Keyword arguments:
        k0 	- Dust absorption coefficiecnt at frequency nu0 (cm^2 g^-1)
        nu0	- Frequency at which k0 is evaluated
        mumol - Mean molecular weight per H2 molecule
    Returns:
        Specific intensity in MJy/sr.
    """
    pars = [nu, NH2, Td]
    parquant = [type(par) == u.Quantity for par in pars]
    if sum(parquant) == 3:
        # Solution for if the main input parameters are astropy quantities
        bb = BlackBody(temperature=Td)
        I_nu = NH2 * mumol * ac.m_p * k0 * (nu / nu0)**beta * bb(nu)
        return I_nu.to('MJy / sr')
    elif sum(parquant) == 0:
        # Solution for if the main input parameters are dimensionless floats
        if type(k0) == u.Quantity:
            k0 = k0.to('cm2 g-1').value
        if type(nu0) == u.Quantity:
            nu0 = nu0.to('GHz').value
        bb = BlackBody(temperature=Td * u.K)
        m_H = ac.m_p.to('g').value
        I_nu = (NH2 * mumol * m_H * k0 * (nu / nu0)**beta *
                bb(nu * 1E9).to('MJy/sr').value)
        return I_nu
    else:
        print('All input parameters must be either u.Quantity or float')
        return np.nan


def highpassimage(image, alpha=0.4, beta=0.3):
    """
    Purpose:
        High pass filter an image.
    Arguments:
        image -	Array to filter
        alpha - The percentage of array values that are tapered.
        beta -  The inner diameter as a fraction of the array size beyond which
                the taper begins. beta must be less or equal to 1.0.
    Returns:
        High-pass filtered image.
    Notes:
    """
    from photutils import SplitCosineBellWindow, TukeyWindow, TopHatWindow

    fftimage = np.fft.fft2(image)
    fftimage2 = np.fft.fftshift(fftimage)

    # Set up the mask
    sizex, sizey = np.shape(fftimage2)
    cenx = (sizex - 1) / 2
    ceny = (sizey - 1) / 2
    Radius = CentredDistanceMatrix(sizex, sizey, cenx, ceny)

# 	mask = np.zeros(np.shape(fftimage))
# 	mask[Radius < scale] = 1.

# 	window = TopHatWindow(beta)
# 	window = TukeyWindow(alpha)
    window = SplitCosineBellWindow(alpha, beta)
    mask = window(fftimage2.shape)

# 	mask *= -1.
# 	mask += 1.#np.zeros(mask.shape)

    fftimage2 *= mask
    fftimage3 = np.fft.ifftshift(fftimage2)
    filteredimage = np.fft.ifft2(fftimage3)

    return (np.abs(filteredimage), mask)


def image(data, wcs, fig=plt.figure(), subplot=111, label=None):
    vmin, vmax = np.nanpercentile(data, [0.5, 99.5])
    ax = fig.add_subplot(subplot, projection=wcs)
    im = ax.imshow(data, vmin=vmin, vmax=vmax)
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_axislabel('Galactic Longitude')
    lat.set_axislabel('Galactic Latitude')
    lon.set_major_formatter('d.d')
    lat.set_major_formatter('d.d')
    lon.set_ticks(spacing=0.5 * u.degree, color='w')
    lat.set_ticks(spacing=0.5 * u.degree, color='w')
    lon.display_minor_ticks(True)
    lat.display_minor_ticks(True)
    lon.set_minor_frequency(5)
    lat.set_minor_frequency(5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="3%", pad=0, axes_class=maxes.Axes)
    cbar = fig.colorbar(im, cax=cax, pad=0)
    cbar.set_label(label)


def index2velocity(index, header):
    """
    Purpose:
        Find the velocity of a given index
    Arguments:
        index - index of velocity plane
        header - header containing wcs information
    Returns:
        Velocity in header units
    """
    crpix = header['CRPIX3']
    crval = header['CRVAL3']
    cdelt = header['CDELT3']
    naxis = header['NAXIS3']
    velocity = cdelt * (index - crpix + 1) + crval
    return velocity


def Jy2K(data, freq, hpbw):
    """
    Converts an astropy quantity from units of Jy (or similar) to K
    """
    beamarea = 2 * np.pi * (hpbw / np.sqrt(8 * np.log(2)))**2
    equiv = u.brightness_temperature(frequency=freq, beam_area=beamarea)
    return data.to(u.K, equivalencies=equiv)


def K2Jy(data, freq, hpbw):
    """
    Converts an astropy quantity from units of K to units of Jy
    """
    beamarea = 2 * np.pi * (hpbw / np.sqrt(8 * np.log(2)))**2
    equiv = u.brightness_temperature(frequency=freq, beam_area=beamarea)
    return data.to(u.Jy, equivalencies=equiv)


def kd_URC(Rgc, R0=8.15, a2=0.96, a3=1.62):
    """
    Universal rotation curve used in Reid et al. (2019) Appendix B.
    The 'A5' fit paramers are used as default
    Input parameters:
        Rgc: Galactocentric radius in kpc
        R0: (=a1) R0 in kpc
        a2: Ropt/R0 where Ropt = 3.2 * R_scalelength encloses 83% of light
        a3: 1.5 * (L/L*)^(1/5)
    Output:
        Tr: Circular rotation speed at Rgc in km/s

    This function is based on Appendix B of Reid et al. (2019), and therefore
    reproduces Table 4 of that paper.
    """
    # The 'Universal rotation curve' of Persic+96, adapted to follow the values
    # used in Reid+16. This should give the implied rotation curve used by
    # Reid+16. Also see
    # http://ned.ipac.caltech.edu/level5/March01/Battaner/node7.html
    # The original equation is listed as 'NOTE ADDED IN PROOF' in Persic+96
    # before the REFERENCES section.
    Lambda = (a3 / 1.5)**5
    Ropt = a2 * R0
    rho = Rgc / Ropt

    log_lam = np.log10(Lambda)
    term1 = 200 * Lambda**0.41

    top = 0.75 * np.exp(-0.4 * Lambda)
    bot = 0.47 + 2.25 * Lambda**0.4
    term2 = np.sqrt(0.8 + 0.49 * log_lam + (top / bot))

    top = 1.97 * rho**1.22
    bot = (rho**2 + 0.61)**1.43
    term3 = (0.72 + 0.44 * log_lam) * (top / bot)

    top = rho**2
    bot = rho**2 + 2.25 * Lambda**0.4
    term4 = 1.6 * np.exp(-0.4 * Lambda) * (top / bot)

    Tr = (term1 / term2) * np.sqrt(term3 + term4)  # km/s
    return Tr


def kd(glon, glat, vlsr, errors=False, ntrials=1E5, R0=8.15, R0err=0.15,
       Omega0=236.7, Omega0err=7, Vsun=11, Vsunerr=0.38, Rmax=20):
    """
    Purpose:
    kd calculates the kinematic distance of a source given its l, b, v
    coordinates using the Persic universal rotation curve, as implemented by
    Reid et al. 2019.

    Input parameters:
        glon: source longitude in degrees
        glat: source latitude in degrees
        vlsr: source velocity in km/s
        errors: return errors?R0 = 8
        ntrials: number of numbers in random sampling to be used for errors
        Omega0: circular rotation speed at Sun's position
        Vsun: global solar motion component in the direction of Gal. rotation
        R0: distance to the Galactic centre
    Output parameters:
           (dnear, dfar): kinematic distance solutions in kpc
        or:
            ((dnear, dfar), (dnear_err, far_err)) if ntrials > 1
    Notes:
        values for Omega0, Vsun and R0 taken from Reid+ 19.
        Vsunerr reverse engineered from:
         omega0 = (Omega0 + Vsun)/R0 = 30.32+-0.27 kms-1 kpc-1
    """
    omega0 = (Omega0 + Vsun) / R0
    Rgc_range = np.arange(0, Rmax, 1E-3)    # 1 pc sampling in Rgc
    Vcirc_range = kd_URC(Rgc_range, R0=R0)
    omega_range = Vcirc_range / Rgc_range
    omega = omega0 + vlsr / (R0 * np.sin(glon * u.deg).value *
                             np.cos(glat * u.deg).value)
    Rgc = Rgc_range[np.nanargmin(np.abs(omega_range - omega))]

    # Kinematic distance quadratic solutions from PhD thesis
    kd_a = np.cos(glat * u.deg).value**2
    kd_b = -2 * R0 * np.cos(glon * u.deg).value * np.cos(glat * u.deg).value
    kd_c = R0**2 - Rgc**2
    kd_far = (-kd_b + np.sqrt(kd_b**2 - 4 * kd_a * kd_c)) / (2 * kd_a)
    kd_near = (-kd_b - np.sqrt(kd_b**2 - 4 * kd_a * kd_c)) / (2 * kd_a)
    if not errors:
        return kd_near, kd_far
    else:
        Omega0_mc = (np.random.normal(Omega0, Omega0err, ntrials))
        Vsun_mc = (np.random.normal(Vsun, Vsunerr, ntrials))
        R0_mc = (np.random.normal(R0, R0err, ntrials))
        omega0_mc = (Omega0_mc + Vsun_mc) / R0_mc
        omega_mc = omega0_mc + vlsr / (R0 * np.sin(glon * u.deg).value *
                                       np.cos(glat * u.deg).value)
        Rgc_mc = np.array([Rgc_range[np.nanargmin(np.abs(omega_range - o))]
                           for o in omega_mc])

        kd_b_mc = (-2 * R0_mc * np.cos(glon * u.deg).value *
                   np.cos(glat * u.deg).value)
        kd_c_mc = R0_mc**2 - Rgc_mc**2
        kd_far_mc = ((-kd_b_mc + np.sqrt(kd_b_mc**2 - 4 * kd_a * kd_c_mc)) /
                     (2 * kd_a))
        kd_near_mc = ((-kd_b_mc - np.sqrt(kd_b_mc**2 - 4 * kd_a * kd_c_mc)) /
                      (2 * kd_a))
        kd_near_err = np.nanstd(kd_near_mc)
        kd_far_err = np.nanstd(kd_far_mc)
        return (kd_near, kd_far), (kd_near_err, kd_far_err)


def linfit(xdata, ydata, x0=[0, 0], sigma=None):
    """
    Purpose:
        Simple linear least squares fit of the form y = m * x + c using
        scipy.optimzation.curve_fit
    Arguments:
        xdata: independent variable
        ydata: dependent variable
        x0 = [0, 0]: a list of guesses for gradient, m, and intercept, c
        sigma: uncertainties on the ydata.
    Returns:
        popt, perr: scipy.optimization.curve_fit instance
    Notes:
        To compute one standard deviation errors on the parameters use
        perr = np.sqrt(np.diag(pcov))
    """
    return optimization.curve_fit(linfunc, xdata, ydata, x0, sigma)


def linfunc(x, m, c):
    return m * x + c


def mad(arr, nan=True):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
        Arguments:
            array
        Outputs:
            The median absolute deviation of a set of numbers
        Options:
            nan=True - by default, exclude nans
    """
    if nan:
        arr = np.ma.array(arr[~np.isnan(arr)]).compressed()
    else:
        arr = np.ma.array(arr).compressed()  # Faster without masked arrays
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def makeheader(pixsize, xcen, ycen, xsize, ysize, system='galactic',
               verbose=False, bunit=''):
    """
    Date added:
                23 March 2018
    Purpose:
                Create a header object. Useful for reprojection, for example.
    Arguments:
        pixsize - pixel size required (arcec)
                xcen, ycen - central coordinates (degrees)
                xsize, ysize - image size (degrees)
    Optional Arguments:
                system - 'radec' or 'galactic' for WCS system
                bunit - set the 'BUNIT' header keyword
    Returns:
                New header
    """
    if system == 'galactic':
        C = SkyCoord(l=xcen * u.degree, b=ycen * u.degree, frame='galactic')
    if system == 'radec':
        C = SkyCoord(ra=xcen * u.degree, dec=ycen * u.degree, frame='icrs')
    datestring = (datetime.datetime.today()).strftime('%Y-%m-%d %X')
    newhdu = fits.PrimaryHDU()  # Initialise HDU with arbitrary 2D array
    hdulist = fits.HDUList([newhdu])
    newheader = hdulist[0].header
    Xsize = int(round_up_to_odd(xsize * 3600 / pixsize))
    Ysize = int(round_up_to_odd(ysize * 3600 / pixsize))
    newheader['NAXIS'] = 2
    newheader['NAXIS1'] = Xsize
    newheader['NAXIS2'] = Ysize
    if system == 'galactic':
        newheader['CTYPE1'] = "GLON-TAN"
        newheader['CTYPE2'] = "GLAT-TAN"
        newheader['CRVAL1'] = C.l.degree
        newheader['CRVAL2'] = C.b.degree
    elif system == 'radec':
        newheader['CTYPE1'] = "RA---TAN"
        newheader['CTYPE2'] = "DEC--TAN"
        newheader['CRVAL1'] = C.icrs.ra.degree
        newheader['CRVAL2'] = C.icrs.dec.degree
    newheader['CRPIX1'] = (Xsize + 1) / 2
    newheader['CRPIX2'] = (Ysize + 1) / 2
    newheader['CDELT1'] = -pixsize / 3600.
    newheader['CDELT2'] = pixsize / 3600.
    newheader['EQUINOX'] = 2000.0
    newheader['LONPOLE'] = 180.0
    newheader['LATPOLE'] = 90.0
    newheader['BUNIT'] = bunit
    newheader['HISTORY'] = 'Header created by ajrpy.makeheader: ' + datestring
    newheader['COMMENT'] = "FITS (Flexible Image Transport System) format is" \
        + " defined in 'Astronomy and Astrophysics', volume 376, page 359; " \
        + "bibcode 2001A&A...376..359H"

    if verbose:
        print("\n	Returning header centered on coodinates %.3f, %.3f" %
              (xcen, ycen))
        print("	Size in degrees: %.3f x %.3f" % (xsize, ysize))
        print("	Size in pixels: %i x %i" % (Xsize, Ysize))
        print("	Central pixel coords: %i, %i" % (newheader['CRPIX1'],
              newheader['CRPIX2']))
        print("	Pixel size: %.3f arcsec" % pixsize)
        print("	System: %s \n" % system)
    return newheader


def mcodr(xdata, xerr, ydata, yerr, ntrials=1000, verbose=False):
    """
    Purpose:
        Calculate a linear fit (y = m*x + c) using orthogonal distance
        regression with uncertainties determined by boostrapping.
    Arguments:
        xdata   - input x data
        xerr    - uncertainties on x data
        ydata   - input y data
        yerr    - uncertainties on y data
        ntrials - number of Monte Carlo trials
    Returns:
        A, Aerr, B, Berr, rho
        Fit coefficients and unceratinties, and a correlation coefficient.
    """
    def function(P, x):
        return P[0] * x + P[1]

    # Initial guess from least-squares fit
    m0, c0, r_value, p_value, std_err = stats.linregress(xdata, ydata)

    Guess = [m0, c0]

    m = np.zeros(ntrials)
    m_err = np.zeros(ntrials)
    c = np.zeros(ntrials)
    c_err = np.zeros(ntrials)
    ndata = len(xdata)

    for i in range(ntrials):
        trialx = np.array([xdata[D] + np.random.normal(0, np.abs(xerr[D])) for
                           D in range(ndata)])
        trialy = np.array([ydata[D] + np.random.normal(0, np.abs(yerr[D])) for
                           D in range(ndata)])
        trialdata = odrpack.RealData(trialx, trialy, sx=xerr, sy=yerr)
        odr = odrpack.ODR(trialdata, odrpack.Model(function), beta0=Guess)
        out = odr.run()
        if verbose:
            out.pprint()
        m[i], c[i] = out.beta
        m_err[i], c_err[i] = out.sd_beta

    sp_rho, sp_p = stat.spearmanr(xdata, ydata)

    meanm = np.nanmean(m)
    meanc = np.nanmean(c)
    stdm = np.nanstd(m)
    stdc = np.nanstd(c)

    return meanm, stdm, meanc, stdc, sp_rho, sp_p


def mJybeam2MJysr(beamfwhm):
    """
    Purpose: Calculate a conversion factor from units of mJy/beam to MJy/sr
             assuming a Gaussian beam, as in Kauffmann et al. (2008)
    Arguments:
        - beamfwhm (arcsec)
    Output:
        - Conversion factor (MJy/sr per mJy/beam)
    """
    omega = 2.665E-11 * beamfwhm**2.
    return 1.E-9 / omega


def MJysr2mJybeam(beamfwhm):
    """
    Purpose: Calculate a conversion factor from MJy/sr to mJy/beam assuming
             gaussian beams
    Arguments:
        - beamfwhm (arcsec)
    Oututs:
        - Conversion factor (mJy/beam per MJy/sr)
    """
    return 1 / mJybeam2MJysr(beamfwhm)


def normal(x, mu, sigma, amp=None):
    """
    Returns a normal distribution of given mean, standard deviation and
    (optionally) amplitude.

    Parameters
    ----------
    x : list or numpy array for the 'x' values
    mu : float - mean value of the distribution
    sigma : flloat - standard deviation
    amp : if given, gives the amplitude of the distribution. If not given,
          the distribution is normalised such that its integral is 1. [None]
    """
    term1 = 1 / (sigma * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * ((x - mu) / sigma)**2)
    if amp is None:
        return term1 * term2
    else:
        return amp * term2


def nrtau(T1, T2, X12):
    """
    Date added:
        (Ancient history)
    Purpose:
        Calculate an LTE optical depth from two lines
    Arguments:
        - T1; intensity of more abundant tracer
        - T2; intensity of less abundant tracer
        - X12; abundance ratio of tracer 1 to 2
    """
    # Initial estimate of optical depth, assuming T1 is optically thick
    Tau0 = -X12 * np.log(1. - (T2 / T1))
    # Return nan values for dud input
    if T1 == np.nan or T2 == np.nan:
        print("	nan")
        return np.nan
    if T1 < 0. or T2 < 0.:
        print(" nan")
        return np.nan
    if T2 > T1:
        return np.nan
    iteration = 0
    ratio = T1 / T2
    agreement = 100
    while (agreement > 0.001):  # Until adjacent iterations agree to 0.1%
        Tau = Tau0 - (((ratio * (1. - np.exp(-Tau0 / X12)))
                      - (1 - np.exp(-Tau0))) / ((ratio * np.exp(-Tau0 / X12)
                                                / X12) - np.exp(-Tau0)))
        agreement = (np.sqrt((Tau - Tau0)**2) / Tau0)
        Tau0 = Tau
        iteration = iteration + 1
        if iteration == 20:
            print(" does not converge")
            break
            return np.nan
    return Tau


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of
    variables in C, controlling for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a
        variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j]
        controlling for the remaining variables in C.

    -------

        Partial Correlation in Python (clone of Matlab's partialcorr)
    This uses the linear regression approach to compute the partial
    correlation (might be slow for a huge number of variables). The
    algorithm is detailed here:
        http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    Taking X and Y two variables of interest and Z the matrix with all the
    variable minus {X, Y},
    the algorithm can be summarized as
        1) perform a normal linear least-squares regression with X as the
           target and Z as the predictor
        2) calculate the residuals in Step #1
        3) perform a normal linear least-squares regression with Y as the
           target and Z as the predictor
        4) calculate the residuals in Step #3
        5) calculate the correlation coefficient between the residuals from
           Steps #2 and #4; The result is the partial correlation between X and
           Y while controlling for the effect of Z
    Date: Nov 2014
    Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    Testing: Valentina Borghesani, valentinaborghesani@gmail.com
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def planck(nu, T):
    """
    Date added:
        6th September 2016
    Purpose:
        Return the Planck black-body function
    Arguments:
        p_nu    - Frequency at which to calculate intensity
        p_t     - Temperature of the black body
    Returns:
        Specific intensity at the given frequency in units of MJy/sr
    """
    # In units of W sr-1 m-2 Hz-1
    P = ((2 * sc.h * nu**3) / (sc.c**2)) * \
        (1 / (np.exp(sc.h * nu / (sc.k * T)) - 1))
    P /= 1E-26     # To convert to Jy sr-1
    P /= 1E6       # To convert to MJy sr-1, /1E-26 / 1E6
    return P


def Planck(freq, temp, unit='Jy'):
    """
    Calculate the Planck function using astropy units
    """
    B = ((2 * ac.h * freq**3) / ac.c**2) \
        * (1 / (np.exp(ac.h * freq / (ac.k_B * temp)) - 1))
    if type(freq) == u.Quantity:
        return B.to(unit) / u.sr
    else:
        return (B.to(unit) / u.sr).value


def pointflux(Mass, freq, Td=15, beta=2, kappa0=0.12, freq0=1.2E12, d=1000,
              verbose=True):
    """
    Date added:
        9th March 2018
    Purpose:
        Calculate the flux density for a given point source mass
    Arguments:
        - Mass [Msol]
        - Freq [Hz]
        - Dust temperature [K, default = 15 K]
        - Dust emissivity spectral index, beta [default = 2]
        - Dust opacity normalisation, kappa0 [cm2 g-1, default = 0.12 cm2 g-1]
        - Distance [pc]
    Returns:		Mass per beam of a point source with those properties
    """
    if type(Mass) == u.quantity.Quantity:
        Mass = Mass.to('kg').value
    if type(freq) == u.quantity.Quantity:
        freq = freq.to('Hz').value
    if type(Td) == u.quantity.Quantity:
        Td = Td.to('K').value
    if type(kappa0) == u.quantity.Quantity:
        kappa0 = kappa0.to('cm2 g-1').value
    if type(d) == u.quantity.Quantity:
        d = d.to('pc').value
    if type(freq0) == u.quantity.Quantity:
        freq0 = freq0.to('Hz').value
    d *= 3.086E18  # Convert pc to cm
    P = planck(freq, Td) * 1.E6  # Convert MJy to Jy
    kappa = kappa0 * (freq / freq0)**beta
    Flux = Mass * (kappa * P) * (Msol * 1000) / d**2
    if verbose:
        print(" Point-mass sensitivity calculation assuming:")
        if Flux < 0.1E-3:
            print("   Flux sensitivity = {:.2f} uJy".format(Flux * 1.E6))
        elif Flux < 0.1:
            print("    Flux sensitivity = {:.2f} mJy".format(Flux * 1.E3))
        else:
            print("    Flux sensitivity = {:.1f} Jy".format(Flux))
        print("     Frequency = %.1f GHz" % (freq / 1.E9))
        print("     Wavelength = %.3f mm" % (1.E3 * sc.c / freq))
        print("     Temperature = %.2f K" % Td)
        print("     Beta = %.2f" % beta)
        print("     Dust opacity = %.2e cm2 g-1" % kappa)
        print("     Dust absorption coefficient = %.2f cm2 g-1 at" % kappa0)
        print("     Reference frequency = %.f GHz" % (freq0 / 1.E9))
        print("     Reference wavelength = %.3f mm" % (1.E3 * sc.c / freq0))
        print("     Distance = %.f pc" % (d / 3.086E18))
        print("     Dust mass = {:.3f} Msol\n".format(Mass))
    return Flux


def pointmass(Flux, freq, Td=15, beta=2, kappa0=0.12, freq0=1.2E12, d=1000,
              verbose=True):
    """
    Date added:
        9th March 2018
    Purpose:
        Calculate the point source mass for a given flux density
    Arguments:
        - Flux density [Jy]
        - Freq [Hz]
    Keyword arguments:
        - Dust temperature [K, default = 15 K]
        - Dust emissivity spectral index, beta [default = 2]
        - Dust opacity normalisation, kappa0 [cm2 g-1, default = 0.12 cm2 g-1]
        - Reference frequency for kappa0
        - Distance [pc]
    Notes:
        - The kappa0 value assumes a gas-to-dust mass ratio of 100.
        - References: Hildebrand 1983 Table 1 for kappa0
    Returns:
        Mass per beam of a point source with those properties
    """
    if type(Flux) == u.quantity.Quantity:
        Flux = Flux.to('Jy').value
    if type(freq) == u.quantity.Quantity:
        freq = freq.to('Hz').value
    if type(Td) == u.quantity.Quantity:
        Td = Td.to('K').value
    if type(kappa0) == u.quantity.Quantity:
        kappa0 = kappa0.to('cm2 g-1').value
    if type(d) == u.quantity.Quantity:
        d = d.to('pc').value
    if type(freq0) == u.quantity.Quantity:
        freq0 = freq0.to('Hz').value
    d *= 3.086E18  # Convert pc to cm
    P = planck(freq, Td) * 1.E6  # Convert MJy to Jy
    kappa = kappa0 * (freq / freq0)**beta
    Mass = d**2 * Flux / (kappa * P) / (Msol * 1000.)
    if verbose:
        print("\n  Pointlike dust mass calculation assuming:")
        if Flux < 0.1E-3:
            print("    Flux sensitivity = {:.2f} uJy".format(Flux * 1.E6))
        elif Flux < 0.1:
            print("    Flux sensitivity = {:.2f} mJy".format(Flux * 1.E3))
        else:
            print("    Flux sensitivity = {:.1f} Jy".format(Flux))
        print("     ========================")
        print("     Frequency = {:.1f} GHz".format(freq / 1.E9))
        print("     Wavelength = {:.3f} mm".format(1.E3 * sc.c / freq))
        print("     Temperature = {:.2f} K".format(Td))
        print("     Beta = {:.2f}".format(beta))
        print("     kappa = {:.2e} cm2 g-1".format(kappa))
        print("     kappa_0 = {:.2f} cm2 g-1 at".format(kappa0))
        print("     Reference frequency = {:.1f} GHz".format(freq0 / 1.E9))
        print("     Reference wavelength = {:.3f} mm"
              .format(1.E3 * sc.c / freq0))
        print("     Distance = {:.1f} pc".format(d / 3.086E18))
        print("     ===========================")
        print("     Dust mass = {:.3f} Msol\n".format(Mass))
    return Mass


def powerspace(start, stop, power, num=10):
    """
    Purpose:
        Generates levels (e.g. for contours) between two limits, given a power
        index.
    Arguments:
        start - the level of the first value
        stop - the level of the final value
        power - the power to which the contours should be spaced
        num - number of levels to generate
    """
    x = start + (np.linspace(0**(1 / power),
                             (stop - start)**(1 / power),
                             num))**power
    return x


def PowerSpectrum(data, header, unit='arcmin', nbins=100):
    """
    Purpose:
        Calculates the FFT of an image, and returns power as a function of
        spatial scale.
    Arguments:
        data   - Array from which to caclulate the power spectrum. Should not
                 contain nans, or else output will be all nans.
        header - (Optional) image header containing pixel size information
        scale  - (Optional)
    Output:
        profile   - The power in spatial scale bins
        spatialscale - power spectrum bin centres

    """

    data[np.isnan(data)] = 0
    power = np.abs(fft.fftshift(fft.fft2(data))**2)
    sizey, sizex = power.shape

    pixsize = np.abs(header['CDELT1']) * u.deg

    frequencies = (CentredDistanceMatrix(sizey, sizex)
                   / (sizex * pixsize))

    edges = np.linspace(0, np.max(frequencies[int((sizey - 1) / 2)]), nbins)
    centres = 0.5 * (edges[0:-1] + edges[1:])

    wavenumber = np.array([])
    profile = np.array([])

    if type(frequencies) == u.Quantity:
        wavenumber *= frequencies.unit
    if type(power) == u.Quantity:
        profile *= power.unit

    for i in range(len(centres) - 1):
        pixels = np.where((frequencies > centres[i])
                          & (frequencies < centres[i + 1]))
        X = np.nanmean(frequencies[pixels])
        Y = np.nanmean(power[pixels])
        wavenumber = np.append(wavenumber, X)
        profile = np.append(profile, Y)

    spatialscale = wavenumber**-1
    return profile, spatialscale.to(unit)


def radex_input(infile, outfile, 
                molecule, freqmin, freqmax, tkin, tbg, 
                voldens, coldens, linewidth):
    """
    Purpose:
        Write an input file for RADEX
    Arguments:
        infile - name of the text file to read in
        outfile - name of the text file to write to
        mol - molecule to analyse and matching a data file in $RADEX_DIR/data
        tkin - gas temperature [K]
        nh2 - H2 volume density (cm-3)
        cdmol - column density of the molecule [cm-2]
        dv - linewidth [km s-1]
        tbg - background temperature [2.73 K]
        fmin - list only lines above this minimum frequency [GHz]
        fupp - list only lines below this maximum frequency [GHz]
    """
    with open(infile, 'w') as f:
        f.write(f'{molecule}.dat\n')
        f.write(f'{outfile}\n')
        f.write(f'{freqmin} {freqmax}\n')
        f.write(f'{tkin}\n')
        f.write('1\n')
        f.write('H2\n')
        f.write(f'{voldens:.3e}\n')
        f.write(f'{tbg}\n')
        f.write(f'{coldens:.3e}\n')
        f.write(f'{linewidth}\n')
        f.write('0\n')  # 0 to tell RADEX to expect no more calculations
        f.close()

def radex_output(outfile, verbose=False,
                 hfs_up="1_0_1", hfs_lo="0_1_2"):
    """
    Purpose: 
        Read the output of RADEX
    Arguments:
        outfile - name of the file containing the output    
        verbose - return the text output for debugging
        hfs_up  - Specific line to return if molecule has hyperfine structure
        hfs_lo  - Specific line to return if molecule has hyperfine structure
                    [nb: default is set to the isolated component 1_0_1-0_1_2]
    Returns:
        quantout - list containing radex quantities of interest:
                t_ex    - excitation temperature
                tau     - optical depth
                t_r     - radiation temperature [K]
                integ   - integrated intensity [K km/s]
    """
    file = open(outfile)
    lines = file.readlines()
    headerline = np.where(["LINE" in line for line in lines])
    rawheader = lines[headerline[0][0]]
    header = np.array(rawheader.split())
    datastart = headerline[0][0] + 2
    datalines = lines[datastart:]
    alldata = [line.split() for line in datalines]
    
    # Check if this is a hyperfine line
    if sum(["hfs" in line for line in lines]) > 0:
        correct_idx = np.where([(hfs_up in dataline) and 
                                (hfs_lo in dataline) 
                                for dataline in alldata])
        data = alldata[correct_idx[0][0]]
    else:
        data = alldata[0]

    startindex = 2
    t_ex = data[np.where(header == "T_EX")[0][0] + startindex]
    tau = data[np.where(header == "TAU")[0][0] + startindex]
    t_r = data[np.where(header == "T_R")[0][0] + startindex]
    integ = data[np.where(header == "FLUX")[0][0] + startindex]

    quantout = [float(out) for out in [t_ex, tau, t_r, integ]]

    if verbose:
        print(rawheader)
        print(datalines)
    
    return quantout


def radex(molecule, freqmin, freqmax, 
          tkin, voldens, coldens, linewidth,
          tbg=2.73,
          infile='temp_radex.inp', 
          outfile='temp_radex.out',
          **kwargs):
    """
    Purpose:
        Run RADEX using the same inputs as the online calculator
    Arguments:
        molecule    - molecule with corresponding RADEX .dat file in place
        freqmin     - min frequency of lines
        freqmax     - max frequency of lines
        tkin        - gas kinetic temperature
        voldens     - volume density of H2 [cm-3]
        coldens     - column density of molecule [cm-2]
        linewidth   - linewidth [km/s]
        tbg         - background temperature [CMB = 2.73 K]
        **kwargs    - keyword arguments for radex_output
    Returns:
        output  - output of radex_output

    """
    radex_input(infile, outfile,
                molecule.lower(), freqmin, freqmax, tkin, tbg, 
                voldens, coldens, linewidth)
    os.system(f'radex < {infile} > /dev/null 2>&1')
    output = radex_output(outfile, **kwargs)

    # Clean up temporary files
    os.system(f'rm {infile}')
    os.system(f'rm {outfile}')
    os.system(f'rm radex.log')

    return output


def rchisq(data, exp, err, ndof):
    """
    Purpose:
        Calculate the reduced chi-squared value of some data
    Arguments:
        data - input data
        exp  - expected values of y
        err  - uncertainty on y
        ndof  - number of degrees of freedom

    Returns:
        Reduced chi-squared value
    Notes:
        Number of freedom is usually: ndof = n - m, where n is the number of
        data points, and m is the number of fitted parameters.
    """
    chisq = (data - exp)**2. / err**2.
    return np.sum(chisq) / ndof


def resolvedmass(Flux, freq, Td=15., beta=2.0, kappa0=0.12, freq0=1.2E12,
                 d=1000., hpbw=15., size=10., verbose=True):
    """
    Date added:
        9th March 2018
    Purpose:
        Calculate the point source mass for a given flux density assuming
        standard values
    Arguments:
        - Flux [Jy]
        - Freq [Hz]
        - Dust temperature [K]
        - Dust beta
        - kappa0 [cm2 g-1]
        - Distance [pc]
    """
    if verbose:
        print(" Resolved dust mass calculation assuming:")
        if Flux < 0.1E-3:
            print("   Flux sensitivity = %.2f uJy" % (Flux * 1.E6))
        elif Flux < 0.1:
            print("    Flux sensitivity = %.2f mJy" % (Flux * 1.E3))
        else:
            print("    Flux sensitivity = %.1f Jy" % Flux)
        print("     Frequency = %.1f GHz" % (freq / 1.E9))
        print("     Wavelength = %.3f mm" % (1.E3 * sc.c / freq))
        print("     Temperature = %.2f K" % Td)
        print("	    Beam FWHM = %.1f arcsec" % hpbw)
        print("	    Source diameter = %.1f arcsec" % size)
        print("     Beta = %.2f" % beta)
        print("     Dust absorption coefficient = %.2f cm2 g-1 at" % kappa0)
        print("     Reference frequency = %.f GHz" % (freq0 / 1.E9))
        print("     Reference wavelength = %.3f mm" % (1.E3 * sc.c / freq0))
        print("     Distance = %.f pc" % d)
    d *= 3.086E18   # Convert pc to cm
    P = planck(freq, Td) * 1.E6 	# Convert MJy to Jy
    kappa = kappa0 * (freq / freq0)**beta
    Mass = d * d * Flux / (kappa * P) / (Msol * 1000.)
    if verbose:
        print("     Dust mass = %.3f Msol" % Mass)
    return Mass


def smoothcube(cube, sigma_smooth):
    """
    Purpose:
        Smooths each plane of a cube
    Arguments:
        cube - cube to be smoothed
        sigma_smooth - sigma of Gaussian smoothing kernel (in pixels)
    Returns:
        Smoothed cube
    """
    cube_out = np.zeros_like(cube)
    kernel = Gaussian2DKernel(sigma_smooth)
    for i in range(np.shape(cube)[0]):
        cube_out[i, :, :] = convolve(cube_in[i, :, :], kernel)
    return cube_out


def smoothimage(image, fwhmin, fwhmout, pixsize, rescale=False, fft=False,
                allow_huge=False, verbose=False):
    """
    Purpose:
    Returns an image smoothed to the desired resolution
    Arguments:
        fwhmin - resolution of input image
        fwhmout - desired smoothed resolution
        pixsize - pixel size.
        rescale -  If true, rescale [per beam] units to account for new
                    effective beamsize (previously 'fixunits')
    Returns:
    Notes:
    fwhmin, fwhmout and pixsize should all have the same units e.g. arcsec
    - Not yet fully consistent with accepting both astropy quantities and
    regular arrays for all arguments...
    """

    sigpix = np.sqrt(fwhmout**2. - fwhmin**2.) * fwhm2sig / pixsize
    if fft:
        from astropy.convolution import Gaussian2DKernel, convolve_fft
        if verbose:
            print('\n   ajrpy.smoothimage: using FFT convolution\n')
        smoothedimage = convolve_fft(image, Gaussian2DKernel(sigpix),
                                     allow_huge=allow_huge)
    else:
        from astropy.convolution import Gaussian2DKernel, convolve
        if verbose:
            print('\n   ajrpy.smoothimage: using standard convolution\n')
        smoothedimage = convolve(image, Gaussian2DKernel(sigpix),
                                 boundary='extend')
    if rescale:
        if type(fwhmin) == u.Quantity:
            smoothedimage *= (fwhmout / fwhmin).value**2
        else:
            smoothedimage *= (fwhmout / fwhmin)**2
    return smoothedimage


def soundspeed(temp, mu=2.8):
    """
    Purpose:
        Returns the isothermal sound speed for gas at a given temperature and
        (optionally) mean particle mass.
    Arguments:
        temp - gas temperature in units of K
        mu - (optional) mean particle mass. Default value of 2.8 for molecular
        gas.
    Returns:
        Sound speed in km/s
    """
    c_s = np.sqrt(ac.k_B * temp * u.K / (mu * ac.m_p)).to('km/s')
    return c_s


def Tcol(flux1, flux2, wav1=160 * u.micron, wav2=250 * u.micron,
         beta=1.8, step=0.1):
    """
    Returns a colour temperature for fluxes at two wavelengths, assuming a
    grey-body SED with a fixed beta.
    Arguments:
        flux1 = flux at wavelength 1
        flux2 = flux at wavelength 2
        wav1 = shortest wavelength (default is 160 micron)
        wav2 = longest wavelength (default is 250 micron)
    Returns:
        Colour temperature in Kelvin.
    """
    if not (type(wav1) == u.Quantity) & (type(wav2) == u.Quantity):
        wav1 *= u.m
        wav2 *= u.m

    Trange = np.arange(2.7, 50.1, step) * u.K
    p1 = np.asarray([Planck(ac.c / (wav1), T).value for T in Trange])
    p2 = np.asarray([Planck(ac.c / (wav2), T).value for T in Trange])
    RatioRange = (p1 / p2) * (wav2 / wav1)**beta
    Tcol = Trange[np.nanargmin(np.abs(flux1 / flux2 - RatioRange))]
    if (Tcol < 2.7 * u.K) or (Tcol > 50 * u.K):
        Tout = np.nan * u.K
    else:
        Tout = Tcol
    if (type(flux1) == u.Quantity) & (type(flux2) == u.Quantity):
        return Tout
    else:
        return Tout.value


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    truncate_colormap

    Truncates a colormap object to a percentile level
    """
    new_cmap = col.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def vaxis(header):
    """
    Purpose: return the velocity axis from a given header
    """
    NAX = header['NAXIS3']
    CDELT = header['CDELT3']
    CRPIX = header['CRPIX3']
    CRVAL = header['CRVAL3']
    vaxis = CDELT * (np.arange(NAX) - CRPIX + 1) + CRVAL

    return vaxis * u.Unit(header['CUNIT3'])



def velocity2index(velocity, header, returnvel=False):
    """
    Purpose:
        Find the index of a given velocity.
    Arguments:

    Returns:
        Includes an option to return the actual velocity for that index.
    """
    crpix = header['CRPIX3']
    crval = header['CRVAL3']
    cdelt = header['CDELT3']
    naxis = header['NAXIS3']
    index = int(crpix - 1 + (velocity - crval) / cdelt)
    truevel = cdelt * (index - crpix + 1) + crval
    if returnvel:
        return index, truevel
    else:
        return index


def wcscutout(map_in, wcs_in, box, frame='galactic'):
    """
    Purpose:
        Cutout a region from a galactic map, and return the cutout and the
        corresponding wcs object
    Arguments:
        map_in  - map to take cutout from
        wcs_in  - corresponding wcs object
        box - [cen1, cen2, size1, size2] list where:
            cen1, cen2 - coordinates of centre point [degrees]
            size1, size2   - width of cutout in axes 1 and 2 [degrees]
        frame - give the wcs frame to use. Default is 'galactic'
    Returns:
        map_cutout, wcs_cutout
    """
    cenl, cenb, sizel, sizeb = box
    cutcen = SkyCoord(cenl * u.degree, cenb * u.degree, frame=frame)
    cutsize = u.Quantity((sizeb, sizel), u.degree)
    cutout = Cutout2D(map_in, cutcen, cutsize, wcs=wcs_in)

    return cutout.data, cutout.wcs


def X12C13CR(Rgc, Rsun=8.5):
    R = Rgc * 8.5 / Rsun    # Normalise for Sun-GC distance
    X = 7.5 * R + 7.6
    Xerr = np.sqrt((R * 1.9)**2. + 12.9**2.)
    return X, Xerr


# ========================= Classes ========================================= #

class MonteCarlo:
    def __init__(self, function, random_numbers, *args_dist_param):

        self.function = function
        self.random_numbers = random_numbers
        self.args_dist_param = args_dist_param
        self.distibution = self.set_normal_distribtuion()

    def set_normal_distribtuion(self):
        self.distibution = rnd.normal
        self.monte_carlo_values = self.do_monte_carlo_error_propagation(
            self.function, self.random_numbers, *self.args_dist_param)

    def set_uniform_distribtuion(self):
        self.distibution = rnd.uniform
        self.monte_carlo_values = \
            self.do_monte_carlo_error_propagation(self.function,
                                                  self.random_numbers,
                                                  *self.args_dist_param)

    def do_monte_carlo_error_propagation(self, function, random_numbers,
                                         *args_dist_param):
        """
        This function calculates Monte Carlo Error Propagation for uniform
        numbers. It take in the function to be evaluated, the amount of random
        numbers and the arguments of the function, as a range between which
        uniform numbers will be drawn from. These are then used to calculate
        the resulting distribution.
        """
        random_numbers = int(random_numbers)
        amount_of_args = int(len(args_dist_param))
        random_numbers_for_func = np.zeros([amount_of_args, random_numbers])

        for arg in range(amount_of_args):
            try:
                distribution_function = args_dist_param[arg][2]

            except IndexError:
                args_dist_param[arg].append(self.distibution)
                distribution_function = args_dist_param[arg][2]

            random_numbers_for_func[arg] = \
                distribution_function(args_dist_param[arg][0],
                                      args_dist_param[arg][1],
                                      random_numbers)

        self.monte_carlo_values = function(*random_numbers_for_func)

        return self.monte_carlo_values

    def mean(self):
        return np.mean(self.monte_carlo_values)

    def median(self):
        return np.median(self.monte_carlo_values)

    def sigma_range(self):

        sorted_values = np.sort(self.monte_carlo_values)
        amount_of_values = len(sorted_values)

        # 0.16 and 0.84 define one standard deviation
        lower_sigma_of_values = sorted_values[int(amount_of_values * 0.16)]
        upper_sigma_of_values = sorted_values[int(amount_of_values * 0.84)]

        return lower_sigma_of_values, upper_sigma_of_values

    def get_all_statistics(self):

        """
        This function gets the mean, median and sigma of a given set of values
        """
        mean_value = self.mean()
        median_value = self.median()
        sigma_of_values = self.sigma_range()

        return mean_value, median_value, sigma_of_values

    def plot_distribution(self):
        """
        This plots the normalised histogram of given values, ploting the mean,
        median and sigma given
        """
        MC_mean, MC_median, MC_sigma = self.get_all_statistics()
        plt.figure('Monte Carlo distribution')
        plt.clf()

        normed_values, bins, patches = plt.hist(self.monte_carlo_values,
                                                bins=100, normed=True,
                                                alpha=0.7)

        # the height of the y axis is the largest normalised value
        height = max(normed_values)
        plt.vlines(x=MC_mean, ymin=0, ymax=height, colors='k', lw=3,
                   label='Mean')

        plt.vlines(x=MC_median, ymin=0, ymax=height, colors='g', lw=3,
                   label='Median')

        plt.fill_between(x=MC_sigma, y1=[height, height], color='r', alpha=0.8,
                         label='1 sigma range')

        plt.legend(loc='best')
        plt.ylim(0, height)
