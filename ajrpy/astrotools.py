
#
# Date: 21 August 2023
#
# Author: Andrew Rigby
#
# Purpose: Contains various useful functions for astronomy applicatins
#

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy import constants as const
from astropy import units as u

import datetime
import numpy as np

import matplotlib as mp
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from regions import Regions
from reproject import reproject_interp, reproject_exact
from scipy.optimize import curve_fit
from sklearn.cluster import AgglomerativeClustering

# ============ Index ======================================================== #
# Constants:
#   fwhm2sig
#   sig2fwhm
# Functions:
#   BeamArea - calculates the area of a Gaussian beam
#   ColBar - wrapper for making tidy colour bars
#   CubeRMS
#   ds9region_to_mask - converts a ds9 region to a binary mask given a header
#   fit_Gaussian
#   fix_imshow_transform - fix imshow issues with reflections from transform
#   friends_of_friends
#   GalacticHeader - creates a 2D Galactic coordinate header, or converts ICRS
#   GalactocentricDistance - calculate RGC given l,b,v or l,b,d
#   get_aspect
#   get_KDA - Return the near/far kinematic distance for give (l, b, RGC)
#   get_KDnear - Return the near kinematic distance for (l, b, v) coodinate
#   get_KDfar - Return the far kinematic distance for (l, b, v) coodinate
#   get_RGC - Calculate RGC given (l, b, v)
#   Gaussian
#   index2vel - converts an array index to a velocity given a cube header
#   Jy2K - converts Jansky per beam to Kelvin
#   K2Jy - converts Kelvin to Jansky per beam
#   planck - returns Planck function as float in units of MJy/sr
#   Planck - returns Planck function as astropy quantity
#   pointflux - calculate the flux density of a point mass
#   pointmass - calculate the point mass for a given flux density
#   reproject_Galactic
#   RMS - returns the root mean square value 
#   RotationCurve - give the value of the rotation velocity given R_GC
#   RoundUpToOdd
#   smoothimage
#   velaxis - generates an array of velocities from a cube header
#   vel2index - converts an array index to a velocity given a cube header
#   wcscutout

# ============ Constants ==================================================== #

fwhm2sig = (8 * np.log(2))**-0.5
sig2fwhm = 1 / fwhm2sig

# ============ Functions ==================================================== #


def BeamArea(fwhm):
    """
    Purpose:
        Give the area of a Gaussian beam with given fwhm

    Arguments:
        Beam FWHM

    Returns:
        Area, in units of units(FWHM)**2. Input as e.g. arcsec or pixels
    """
    return fwhm**2. * np.pi / (4. * np.log(2.))


def ColBar(fig, ax, im, label='', position='right', size="5%",
           dividerpad=0.0, cbarpad=0.15, hide=False, **kwargs):
    """
    Purpose:
        Produces a decent default colour bar attached to the side of an image

    Arguments:
        fig - figure object
        ax - axis object
        im - imshow axis object
        **kwargs - keyword arguments for a fig.colorbar object

    Optional arguments:
        label - (string) label for the colour bar ['']
        size - (string) size of the colorbar as perecentage ["5%"]
        dividerpad - color bar padding [0.2] default from
                     rcParams["figure.subplot.wspace"]
        https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.axes_divider.AxesDivider.html#mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes
        cbarpad - color bad padding [0.15 for vertical color bar]
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
   
    Returns:
        matplotlib.colorbar.Colorbar object
    """
    if (position == 'top') | (position == 'bottom'):
        orientation = 'horizontal'
    else:
        orientation = 'vertical'
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=dividerpad,
                              axes_class=maxes.Axes)
    cbar = fig.colorbar(im, cax=cax, pad=cbarpad,
                        orientation=orientation, **kwargs)
    cbar.set_label(label)
    if (position == 'top') | (position == 'bottom'):
        cax.xaxis.set_ticks_position(position)
        cax.xaxis.set_label_position(position)
    cbar.ax.yaxis.set_tick_params(color='k')
    cbar.ax.minorticks_off()
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='k')
    if hide:
        cbar.remove()

    return cbar


def CubeRMS(cube):
    """
    Purpose:
        Returns an RMS map for a cube containing emission by taking only the
        negative values, and assuming that the noise is normally distributed
        around zero.

    Arguments:
        cube - the data cube for which the RMS map is required
    
    Returns:
        A 2-D array that is the RMS map
    """
    data = cube.copy()
    data[data > 0] = np.nan
    # inverted_data = data * -1
    # combined_data = np.concatenate([data, inverted_data])
    return RMS(data, axis=0)


def ds9region_to_mask(regionfile, header):
    """
    Converts a ds9 region (.reg file) to a 2D mask
    """
    # Read in DS9 region using the 'regions' package
    region = Regions.read(regionfile)[0]

    # Work out how large the image or cube is
    nx = header['NAXIS1']
    ny = header['NAXIS2']

    # Read the WCS information from the header
    wcs = WCS(header)

    # Make array of 2D pixel coordinates
    xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))

    # Convert these to Sky Coordinates
    coords = wcs.celestial.all_pix2world(xv.ravel(), yv.ravel(), 0)
    SkyCoords = SkyCoord(coords[0] * u.deg, coords[1] * u.deg, frame='galactic')

    # Check if the list of pixel coordinates fall within the region
    contains_coords = region.contains(SkyCoords, wcs=wcs.celestial)

    # Reshape the result into 2D image
    mask = contains_coords.reshape(nx, ny) * 1.0
    return mask


def fit_Gaussian(array, mu0=None, std0=None, amp0=None, fixmu=False,
                 return_cov=False):
    if std0 is None:
        std0 = np.nanstd(array)
    if mu0 is None:
        mu0 = np.nanmean(array)
        
    histrange = [-10 * std0, 10 * std0]
    nbins = 1000
    hist, edges = np.histogram(array, range=histrange, bins=nbins)
    if amp0 is None:
        amp0 = hist.max()
    centres = 0.5 * (edges[:-1] + edges[1:])
    if fixmu:
        popt, pcov = curve_fit(lambda x, a: Gaussian(x, 0, std0, amp0), centres, hist)
    else:
        popt, pcov = curve_fit(Gaussian, centres, hist, p0=[0, std0, hist.max()])
    if return_cov:
        return popt, pcov
    else:
        return popt


def fix_imshow_transform(im, ax, wcs):
    """
    Performs a fix to an axis object where an image is being shown with a 
    different wcs than the axes using a transform. If the image has a border of
    nans, the default behaviour results in strange additional images.

    This fix should work OK for small-scale images, where sky distortions are
    minimal, otherwise pcolormesh should be used. See:
        https://github.com/astropy/astropy/issues/14420
              
    Arguments:
            im - an imshow object
            ax - the axis object to which the imshow was added
            wcs - the wcs of image to transform
    """
    data = im.get_array().data
    w = data.shape[1]
    h = data.shape[0]
    path = mp.path.Path([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], 
                         [-0.5, h - 0.5], [-0.5, -0.5]])
    im.set_clip_path(path, transform=ax.get_transform(wcs))
    

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


def GalacticHeader(coords_in=None, header_in=None, frame_in='icrs'):
    """
    Purpose: 
        Create a simple 2D header in Galatic Coordinates. You can either
        specify coords_in, or provide an equatorial header as header_in, which
        will be converted to a Galactic equivalent.

    Arguments:
        coords_in: A tuple (cenL, cenB, sizeL, sizeB, pixsize, bunit) where:
            cenL    = Central Galactic Longitude in degrees
            cenB    = Central Galactic Latitude in degrees
            sizeL   = Array size in Longitude axis in degrees
            sizeB   = Array size in Latitude axis in degrees
            pixsize = pixel size in arcseconds
            bunit   = unit of the intensity scale
        header_in: If supplied with a header in equatorial coorinates,
                   return a similar header in Galactic coordinates
        frame_in: If using header_in, specify frame_in, if the frame is not 
                  ICRS. This is usually stored in the 'RADESYS' header card.
    
    Returns:
        Header in Galactic coordinates

    Examples:
        1) Create a Galactic header from scratch:

        import ajrpy.astrtools as at
        
        newheader = GalacticHeader(coords_in=(23.9, 0.0, 
                                              3.0, 2.0, 
                                              3.0, 'Jy/beam'))

        2) Convert an existing Equatorial header

        import ajrpy.astrtools as at
        from astropy.io import fits

        equatorial_header = fits.getheader('equatorial_image.fits')
        galheader = GalacticHeader(header_in=equatorial_header, frame_in='FK5')
            
    """
    if (coords_in is not None) & (header_in is None):
        cenL, cenB, sizeL, sizeB, pixsize, bunit = coords_in
        centre = SkyCoord(l=cenL * u.degree, b=cenB * u.degree, frame='galactic')
        newhdu = fits.PrimaryHDU()  # Initialise HDU with arbitrary 2D array
        hdulist = fits.HDUList([newhdu])
        newheader = hdulist[0].header
        Lsize = int(RoundUpToOdd(sizeL * 3600 / pixsize))
        Bsize = int(RoundUpToOdd(sizeB * 3600 / pixsize))
        newheader['NAXIS'] = 2
        newheader['NAXIS1'] = Lsize
        newheader['NAXIS2'] = Bsize
        newheader['BITPIX'] = -32
        newheader['CTYPE1'] = "GLON-TAN"
        newheader['CTYPE2'] = "GLAT-TAN"
        newheader['CRVAL1'] = centre.l.degree
        newheader['CRVAL2'] = centre.b.degree
        newheader['CRPIX1'] = (Lsize + 1) / 2
        newheader['CRPIX2'] = (Bsize + 1) / 2
        newheader['CDELT1'] = -pixsize / 3600.
        newheader['CDELT2'] = pixsize / 3600.
        newheader['EQUINOX'] = 2000.0
        newheader['LONPOLE'] = 180.0
        newheader['LATPOLE'] = 90.0
        newheader['BUNIT'] = bunit
        newheader['COMMENT'] = "FITS (Flexible Image Transport System) format is" \
            + " defined in 'Astronomy and Astrophysics', volume 376, page 359; " \
            + "bibcode 2001A&A...376..359H"
    elif (coords_in is None) & (header_in is not None):
        newheader = header_in.copy()
        cenRA = header_in['CRVAL1'] * u.deg
        cenDec = header_in['CRVAL2'] * u.deg
        centre = SkyCoord(cenRA, cenDec, frame=frame_in)
        newheader['CRVAL1'] = centre.galactic.l.value
        newheader['CRVAL2'] = centre.galactic.b.value
        newheader['CTYPE1'] = header_in['CTYPE1'].replace('RA--', 'GLON')
        newheader['CTYPE2'] = header_in['CTYPE2'].replace('DEC-', 'GLAT')
    else:
        raise Exception('Must give either coords_in or header_in, not both')

    datestring = (datetime.datetime.today()).strftime('%Y-%m-%d %X')
    newheader['HISTORY'] = 'Header created by ajrpy.astrotools.GalacticHeader '
    newheader['HISTORY'] = 'Header created on ' + datestring

    return newheader


def GalactocentricDistance(glon, dist=None, vlsr=None, 
                           R0=8.15 * u.kpc, theta0=236 * u.km / u.s):
    """
    Return the Galactocentric distance of a source given l and either
    distance or velocity. Assumes b is zero (or close enough).

    Arguments:
        glon
      # glat
        dist
        vlsr

    Returns:
        Galactocentric distance in kpc

    """
    glon *= u.deg
    vlsr *= u.km / u.s
    if dist is not None:
        # Cosine rule
        RGC = np.sqrt(R0**2 + dist**2 - 2 * R0 * dist * np.cos(glon))
    elif vlsr is not None:
        # Equation 3 from Binney & Tremaine (1991):
        # v_r = [theta(R) / R + theta0/R0] * R0 * sin(glon)
        #
        # Rearranging we get:
        # v_r / (R0 * sin(glon)) + theta0/R0 = theta(R) / R
        #
        Rrange = np.linspace(0, 50, 10001) * u.kpc
        Rvels = (np.array([RotationCurve(r.value).value for r in Rrange]) * 
                 u.km / u.s)
        RHS = Rvels / Rrange
        
        # Rearranging equation 3 of Binney & Tremaine 1991
        LHS = vlsr / (R0 * np.sin(glon)) + theta0 / R0

        RGC = Rrange[np.nanargmin(np.abs(LHS - RHS))]
    else:
        raise Exception('Must give either dist or vlsr')

    return RGC



def Gaussian(x, mu, sigma, amp=None):
    """
    Returns a Gaussian distribution of given mean, standard deviation and
    (optionally) amplitude.

    Arguments:
        x     : float - list or numpy array for the 'x' values
        mu    : float - mean value of the distribution
        sigma : float - standard deviation
        amp   : float - if given, gives the amplitude of the distribution. If 
                not given, the distribution is normalised.

    Returns:
        Gaussian distribution. float.
    """
    term1 = 1 / (sigma * np.sqrt(2 * np.pi))
    term2 = np.exp(-0.5 * ((x - mu) / sigma)**2)
    if amp is None:
        return term1 * term2
    else:
        return amp * term2
    

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


def get_KDA(glon, glat, RGC, R0=8.15 * u.kpc):
    """
    Determine the near and far kinematic distances from the quadratic formula
    following equation 4.9 from my thesis.

    Arguments:
        glon - Galactic longitude (as astropy quantity)
        glat - Galactic latitude (as astropy quantity)
        RGC - Galactocentric radius (as astropy quantity)

    Returns:
        Astropy quantity array containing [near distance, far distance]
    """

    cosb = np.cos(glat)
    cosl = np.cos(glon)

    RR = 2 * R0 * cosl * cosb

    # Discriminant
    dd = 4 * R0**2 * cosl**2 * cosb**2 - 4 * cosb**2 * (R0**2 - RGC**2)

    if dd.value < 0:
        print('Negative discriminant in kinematic distance formula.')
        print('Returning tangent.')
        dT = RR / (2 * cosb**2)
        return np.array([dT.value, dT.value]) * dT.unit
    else:
        dN = (RR - np.sqrt(dd)) / (2 * cosb**2)  # Near distance
        dF = (RR + np.sqrt(dd)) / (2 * cosb**2)  # Far distance

    return np.array([dN.value, dF.value]) * dN.unit


def get_KDnear(glon, glat, vlsr):
    """
    Wrapper to get the near kinematic distance for a given l, b, v coordinate
    Arguments:
        glon - Galactic longitude (as astropy quantity)
        glat - Galactic latitude (as astropy quantity)
        vlsr - Radial velocity (as astropy quantity)
    Returns:
        Near kinematic distance in kpc
    """
    RGCi = get_RGC(glon, glat, vlsr)

    return get_KDA(glon, glat, RGCi)[0]


def get_KDfar(glon, glat, vlsr):
    """
    Wrapper to get the far kinematic distance for a given l, b, v coordinate
    Arguments:
        glon - Galactic longitude (as astropy quantity)
        glat - Galactic latitude (as astropy quantity)
        vlsr - Radial velocity (as astropy quantity)
    Returns:
        Far kinematic distance in kpc
    """
    RGCi = get_RGC(glon, glat, vlsr)

    return get_KDA(glon, glat, RGCi)[1]


def get_RGC(glon, glat, vlsr, 
            R0=8.15 * u.kpc,
            omega0 = 30.32 * u.km / u.s / u.kpc):
    """
    Calulate the Galactocentric radius for an l, b, v coordinate

    Arguments:
        glon
        glat
        vlsr
    Returns:
        Galactocentric radius to the nearest 10 pc

    Notes:
        - Follows Equation 4.7 from my thesis.
        - Values for R0, omega0 from Reid+19
        - Sampled range of RGC are between 0 and 30 kpc.
    """

    RGC = np.arange(0, 30, step=1E-2)[1:] * u.kpc
    rotcurve = [RotationCurve(rad.value).value for rad in RGC]
    vcirc = np.array(rotcurve) * u.km / u.s
    OmegaCurve = vcirc / RGC
    omega = omega0 + vlsr / (R0 * np.sin(glon) * np.cos(glat))
    diff = np.abs(OmegaCurve - omega)
    index = np.nanargmin(diff)
    best_RGC = RGC[index]

    return best_RGC.to('kpc')


def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    Source:
    https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

x = np.random.randn(100, 100, 100)
hessian(x)
    

def index2vel(index, header):
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
    velocity = cdelt * (index - crpix + 1) + crval
    return velocity


def Jy2K(data, freq, hpbw):
    """
    Purpose:
        Converts an astropy quantity from units of Jy (or similar) to K
    Arguments:
        data - data value (in u.Jy or equivalent)
        freq - frequncy of observations (in u.Hz or equivalent)
        hpbw - half power beamwidth (in u.deg or equivalent)

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


def OH5(wav, log=True, scaledown=1.0):
    """
    Purpose:
        Returns a value of Ossenkopf & Henning (1994) dust opacity from Table 1 column 5.
    Arguments:
        wav       - wavelength (in um) at which to return kappa
        log       - if True, interpolate in log-space, or else interpolate 
                    linearly
        scaledown - scale the resulting kappa by this factor. Kauffmann+10 
                    suggest a value of 1.5. Schuller et al. 2009 do not use 
                    this for ATLASGAL, so a default value of 1.0 is used here 
                    to not scale down.
    """
    wavelength0 = np.array([54.1, 63.1, 73.6, 85.8, 
                           100, 117, 136, 158, 185, 
                           226, 350, 500, 700, 1000, 1300])
    kappa0 = np.array([2.8E2, 2.11E2, 1.58E2, 1.18E2, 
                      8.65E1, 6.75E1, 5.25E1, 4.09E1, 3.07E1, 
                      2.17E1, 1.01E1, 5.04E0, 2.57E0, 1.37E0, 8.99E-1])
    if log:
        kappa = 10**np.interp(np.log10(wav), np.log10(wavelength0), 
                            np.log10(kappa0))
    else:
        kappa = np.interp(wav, wavelength0, kappa0)
    return kappa / scaledown



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
    P = ((2 * const.h.value * nu**3) / (const.c.value**2)) * \
        (1 / (np.exp(const.h.value * nu / (const.k_B.value * T)) - 1))
    P /= 1E-26     # To convert to Jy sr-1
    P /= 1E6       # To convert to MJy sr-1, /1E-26 / 1E6
    return P


def Planck(freq, temp, unit='Jy'):
    """
    Calculate the Planck function using astropy units
    """
    if type(freq) == u.Quantity:
        freq = freq.to('Hz')
    else:
        freq *= u.Hz
    if type(temp) == u.Quantity:
        temp = temp.to('K')
    else:
        temp *= u.K
    B = ((2 * const.h * freq**3) / const.c**2) \
        * (1 / (np.exp(const.h * freq / (const.k_B * temp)) - 1))
    if type(freq) == u.Quantity:
        return B.to(unit) / u.sr
    else:
        return (B.to(unit) / u.sr).value
    

def pointflux(Mass, freq, d=1 * u.kpc, Td=15 * u.K, beta=2, 
              g2d=100, kappa0=0.12 * u.cm**2 / u.g, freq0=1200 * u.GHz,
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
    if type(Mass) != u.quantity.Quantity:
        Mass *= u.Msun
    if type(freq) != u.quantity.Quantity:
        freq *= u.GHz
    if type(Td) != u.quantity.Quantity:
        Td *= u.K
    if type(kappa0) != u.quantity.Quantity:
        kappa0 *= u.cm**2 / u.g
    if type(d) != u.quantity.Quantity:
        d *= u.pc
    if type(freq0) != u.quantity.Quantity:
        freq0 *= u.GHz   

    wav = (const.c / freq).to('mm')
    wav0 = (const.c / freq0).to('mm')
    kappa = (kappa0 * (freq / freq0)**beta).to('cm2 g-1')
    P = Planck(freq, Td) * u.sr
    flux = (Mass * kappa * P / d**2).to('mJy')

    if flux.value > 1000:
        flux.to('Jy')
    if flux.value < 10:
        flux.to('uJy')

    if verbose:
        print(" Point-mass sensitivity calculation assuming:")
        print(f"     Flux sensitivity = {flux:.1f}")
        print(f"     Frequency = {freq:.0f}")
        print(f"     Wavelength = {wav:.2f}")
        print(f"     Temperature = {Td:.1f}")
        print(f"     Beta = {beta:.1f}")
        print(f"     Dust opacity = {kappa:.2e}")
        print(f"     Dust absorption coefficient = {kappa0:.2f} at")
        print(f"     Reference frequency = {freq0}")
        print(f"     Reference wavelength = {wav0:.3f}")
        print(f"     Distance = {d:.3f}")
        print(f"     Dust mass = {Mass:.3f}")

    return flux


def pointmass(flux, freq, d=1 * u.kpc, Td=15 * u.K, beta=2, 
              g2d=100, kappa0=12.0 * u.cm**2 / u.g, freq0=1.2 * u.THz,
              verbose=True,):
    """
    Date added:
        9th March 2018
    Purpose:
        Calculate the point source mass for a given flux density
    Arguments:
        - flux: Flux density [Jy]
        - freq: Frequency [GHz]
    Keyword arguments:
        - beta: Dust emissivity spectral index [default = 2]
        - Dust temperature [K, default = 15 K]
        - Dust opacity normalisation, kappa0 [cm2 g-1, default = 0.12 cm2 g-1]
        - Reference frequency for kappa0
        - Distance [pc]
    Notes:
        - The kappa0 value assumes a gas-to-dust mass ratio of 100.
        - References: Hildebrand 1983 Table 1 for kappa0
    Returns:
        Mass per beam of a point source with those properties
    """
    if type(flux) != u.quantity.Quantity:
        flux *= u.Jy
    if type(freq) != u.quantity.Quantity:
        freq *= u.GHz
    if type(Td) != u.quantity.Quantity:
        Td *= u.K
    if type(kappa0) != u.quantity.Quantity:
        kappa0 *= u.cm**2 / u.g
    if type(d) != u.quantity.Quantity:
        d *= u.pc
    if type(freq0) != u.quantity.Quantity:
        freq0 *= u.GHz   
    wav = (const.c / freq).to('mm')
    wav0 = (const.c / freq0).to('mm')
    kappa = (kappa0 * (freq / freq0)**beta).to('cm2 g-1')
    Mass = ((d**2 / u.sr)  * flux * g2d / (kappa * Planck(freq, Td))).to('Msun')
    if verbose:
        print("\n     Point-like dust mass calculation assuming:")
        print("     ===============================")
        if flux < 0.1 * u.mJy:
            print(f"     Flux = {flux.to('uJy')}")
        elif flux < 0.1 * u.Jy:
            print(f"     Flux = {flux.to('mJy')}")
        print(f"     Frequency = {freq.to('GHz'):.1f}")
        if wav < 1 * u.mm:
            print(f"     Wavelength = {wav.to('um'):.3f}")
        else:
            print(f"     Wavelength = {wav.to('mm'):.3f}")
        print(f"     Temperature = {Td:.2f}")
        print(f"     Beta = {beta:.2f}")
        print(f"     gas to dust mass ratio = {g2d}")
        print(f"     kappa = {kappa.to('cm2 g-1'):.5f}")
        print(f"     kappa_0 = {kappa0:.2f} at")
        print(f"     Reference frequency = {freq0.to('GHz'):.1f}")
        print(f"     Reference wavelength = {wav0:.3f}")
        print(f"     Distance = {d:.1f}")
        print("     ===============================")
        print(f"     Dust mass = {Mass:.3f}\n")
    return Mass


def reproject_Galactic(file_in, file_out=None, method="exact", write=True,
                       verbose=False, ext=0, trim=True, **kwargs):
    """
    Purpose:
        Reproject an image from Equatorial to Galactic coordinates using the 
        reproject package: https://reproject.readthedocs.io/en/stable/index.html

    Arguments:
        file_in   - string. Name of the .fits file to reproject
        file_out  - string. Specify the output filename. If not specified, the
                    file will be written to the same location as the input file
                    but with the suffix "_GAL"
        method    - string. The reprojection method. Currently available options
                    are either "exact" or "interp". The "exact" method is slower,
                    and the difference between the two is generally minimal, so
                    if the reproject is taking a long time, the "interp" method
                    could be used. Default is "exact".
        verbose   - boolean. Produces some print statements.
        write     - boolean. Save the fits image to file?
        ext       - the fits extension containing the data & header. Assumed to 
                    be extension 0 by default.
        trim      - boolean. Trim any border of nan-valued pixels from the array
        **kwargs  - keyword arguments passed to the reprojection method.
    
    Returns:
        reprojected_data - Array containing the reprojected data
        header_out       - The Galactic header
        
        Also writes out the reprojected image to a .fits file if write=True.
    """

    if file_out is None:
        file_out = file_in.replace('.fits', '_GAL.fits')

    hdu_in = fits.open(file_in)
    header_in = hdu_in[ext].header
    header_out = GalacticHeader(header_in=header_in)

    if method == "interp":
        if verbose:
            print('\nReprojecting using reproject_interp...')
        reprojected_data, _ = reproject_interp(hdu_in, header_out)
    elif method == "exact":
        if verbose:
            print('\nReprojecting using reproject_exact...')
        reprojected_data, _ = reproject_exact(hdu_in, header_out)
    else:
        raise Exception("Reproject method not recognised. Must be either 'exact' or 'interp'.")
    
    if trim:
        if verbose:
            print('Trimming any nan padding from array')
        mask = 1 * ~np.isnan(reprojected_data)
        axisX = np.sum(mask, axis=0)
        axisY = np.sum(mask, axis=1)
        minY = np.where(axisY > 0)[0][0]
        maxY = np.where(axisY > 0)[0][-1]
        minX = np.where(axisX > 0)[0][0]
        maxX = np.where(axisX > 0)[0][-1]

        cropped_data = reprojected_data[minY:maxY + 1, minX:maxX + 1]

        # Update the header accordingly
        header_out['NAXIS1'] = cropped_data.shape[1]
        header_out['NAXIS2'] = cropped_data.shape[0]
        header_out['CRPIX1'] -= minX
        header_out['CRPIX2'] -= minY

        reprojected_data = cropped_data

    if write:
        # Write to file
        fits.writeto(file_out, reprojected_data, header_out, overwrite=True)

    return reprojected_data, header_out


def RMS(array, nan=True, **kwargs):
    """
    Purpose:
        Returns the root-mean-square value of an array

    Arguments:
        Array - list, or numpy array, etc.
        nan - boolean. If true, uses np.nanmean as opposed to np.mean
        **kwargs for np.nanmean or np.mean
    
    Returns:
        RMS value

    Examples:
        To return a 2D RMS map of a data cube:
            > rmsmap = generaltools.rms(cube, axis=0)
        To return the overall RMS value of a data cube:
            > rmsval = generaltools.rms(cube)
    """
    if nan:
        return np.sqrt(np.nanmean(array**2, **kwargs))
    else:
        return np.sqrt(np.mean(array**2, **kwargs))
    

def RotationCurve(RGC, a2=0.96, a3=1.62, R0=8.15):
    """
    Disk plus halo parameterization of the rotation curve of Persic et al. 1996
    Adapted from the FORTRAN routine listed in Appendix B of Reid et al. (2019)
    Arguments:
        RGC - Galactocentric radius in kpc
        a2 - Ropt/Ro where Ropt=3.2*R_scalelength enclosing 83% of light
        a3 - 1.5*(L/Lstar)**(1/5)
        R0 - R0 in kpc
    Returns:
        Circular rotation speed at RGC in km/s
    """
    lam = (a3 / 1.5)**5  # L/Lstar
    Ropt = a2 * R0 
    rho = RGC / Ropt 

    # Calculate Tr... 
    term1 = 200. * lam**0.41

    top = 0.75 * np.exp(-0.4 * lam)
    bot = 0.47 + 2.25 * lam**0.4 
    term2 = np.sqrt(0.80 + 0.49 * np.log10(lam) + (top / bot))
    
    top = 1.97 * rho ** 1.22 
    bot = (rho**2 + 0.61) ** 1.43 
    term3 = (0.72 + 0.44 * np.log10(lam)) * (top / bot) 
    
    top = rho**2 
    bot = rho**2+2.25 * lam**0.4 
    term4 = 1.6 * np.exp(-0.4 * lam) * (top / bot) 
    
    Tr= (term1 / term2) * np.sqrt(term3 + term4) # km/s 
    
    return Tr * u.km / u.s
    

def RoundUpToOdd(f):
    """
    Returns:
        Round a floating point number up to the nearest odd integer
    """
    return np.ceil(f) // 2 * 2 + 1


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


def velaxis(header):
    """
    Purpose: return the velocity axis from a given header
    """
    NAX = header['NAXIS3']
    CDELT = header['CDELT3']
    CRPIX = header['CRPIX3']
    CRVAL = header['CRVAL3']
    vaxis = CDELT * (np.arange(NAX) - CRPIX + 1) + CRVAL

    return vaxis * u.Unit(header['CUNIT3'])


def vel2index(velocity, header, returnvel=False, returnint=True):
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
    index = crpix - 1 + (velocity - crval) / cdelt
    if returnint:
        index = int(index)
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
    assert cenl.unit.is_equivalent(u.deg), "cenl must have astropy angle units"
    assert cenb.unit.is_equivalent(u.deg), "cenb must have astropy angle units"
    assert sizel.unit.is_equivalent(u.deg), "sizel must have astropy angle units"
    assert sizeb.unit.is_equivalent(u.deg), "sizeb must have astropy angle units"

    cutcen = SkyCoord(cenl, cenb, frame=frame)
    cutsize = u.Quantity((sizeb, sizel))
    cutout = Cutout2D(map_in, cutcen, cutsize, wcs=wcs_in)

    return cutout.data, cutout.wcs
