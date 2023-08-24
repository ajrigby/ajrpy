# ajrpy
A module containing generally useful functions for astronomy and data analysis. The package was written in Python 3.10.12, but should be compatible with any Python 3 version.

## Installation
The simplest way to install this package is through pip:

```zsh
pip install git+https://github.com/ajrigby/ajrpy.git
```

## Usage
Most of the useful functions are located within the astrotools module. All functions have a docstring
which allows you to examine their usage, for example:

```python
import ajrpy.astrotools as at

at.GalacticHeader?
```

Returns a doc string like the following:
```
Signature: at.GalacticHeader(coords_in=None, header_in=None, frame_in='icrs')
Docstring:
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
        
File:      ~/Library/CloudStorage/OneDrive-UniversityofLeeds/Projects/Python/ajrpy/ajrpy/astrotools.py
Type:      function
```


In this case, GalacticHeader can be used as follows:

```python
import ajrpy.astrotools as at

# Create an image header in Galactic coordinates based on a tuple with 6 items:
lcen = 23.9
bcen = 0.05
lsize = 3.0 # degrees
bsize = 2.0 # degrees
pixsize = 3.0 # arcsec
bunit = 'mJy/beam' # 'BUNIT' item for header gives pixel units

header = at.GalacticHeader(coords_in=(lcen, bcen, lsize, bsize, pixsize, bunit))
```

## Functions

`BeamArea` - Returns the area of a Gaussian beam with a given FWHM  
`ColBar` - Returns a simple matplotlib.pyplot.figure instance with some useful customization  
`CubeRMS` - Reurns an RMS map from a given cube by inverting the negative pixel values  
`GalacticHeader` - Create a new Galactic header, or convert an existing header to Galactic coordinates  
`reproject_Galactic` - Reprojects an equatorial .fits image onto a pixel grid in Galactic coordinates  
`RMS` - Return the RMS value or array for some input data  
`RoundUpToOdd` - Rounds a value up to the nearest odd integer  
