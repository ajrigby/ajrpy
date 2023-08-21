
#
# Date: 21 August 2023
#
# Author: Andrew Rigby
#
# Purpose: Contains various useful functions for astronomy applicatins
#

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import datetime
import numpy as np
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ============ Index ======================================================== #
# Constants:
# - fwhm2sig
# - sig2fwhm
# Functions:
# - BeamArea
# - ColBar
# - CubeRMS
# - GalacticHeader
# - RMS
# - RoundUpToOdd
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
    Notes:
    """
    return fwhm**2. * np.pi / (4. * np.log(2.))


def ColBar(fig, ax, im, label='', position='right', size="5%",
           dividerpad=0.0, cbarpad=0.15, **kwargs):
    """
    Purpose:
        Produces a decent default colour bar attached to the side of an image
    Arguments:
        fig - figure object
        ax - axis object
        im - imshow axis object
        **kwags - keyword arguments for a fig.colorbar object
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
    return cbar


def CubeRMS(cube):
    """
    Purpose:
        Returns an RMS map for a cube containing emission by inverting the
        negative data values and assuming the noise is normally distributed
        around zero.
    Arguments:
        cube - the data cube for which the RMS map is required
    Returns
        RMSmap
    """
    data = cube.copy()
    data[data > 0] = np.nan
    inverted_data = data * -1
    combined_data = np.concatenate([data, inverted_data])
    return RMS(combined_data, axis=0)


def GalacticHeader(coords_in=None, header_in=None, frame_in='icrs'):
    """
    Purpose: create a header in Galatic Coordinates.
    Arguments:
        coords_in: A tuple containing 6 elements:
            cenL - Central Galactic Longitude in degrees
            cenB - Central Galactic Latitude in degrees
            sizeL - Array size in Longitude axis in degrees
            sizeB - Array size in Latitude axis in degrees
            pixsize - pixel size in arcseconds
            bunit - unit of the intensity scale
        header_in: If supplied with a header in equatorial coorinates,
                   return a similar header in Galactic coordinates
        frame_in: If using header_in, specify frame_in, if the frame is not 
                  ICRS. This is usually stored in the 'RADESYS' header card.
    Returns:
        Header in Galactic coordinates
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
    newheader['HISTORY'] = 'Header created by ajrpy.astrotools ' + datestring

    return newheader





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
    

def RoundUpToOdd(f):
    """
    Purpose:
        Round a floating point number up to the nearest odd integer
    """
    return np.ceil(f) // 2 * 2 + 1