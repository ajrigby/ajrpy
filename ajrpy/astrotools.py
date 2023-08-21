
#
# Date: 21 August 2023
#
# Author: Andrew Rigby
#
# Purpose: Contains various useful functions for astronomy applicatins
#

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import datetime

# ============ Index: ======================================================= #
# GalacticHeader
# RMS
# RoundUpToOdd

# =========================================================================== #

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