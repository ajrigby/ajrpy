# ajrpy
A module containing generally useful functions for astronomy and data analysis. The package was written in Python 3.10.12, but should be compatible with any Python 3 version.

## Installation
The simplest way to install this package is through pip:

```zsh
pip install git+https://github.com/ajrigby/ajrpy.git
```

## Usage
Most of the useful functions are located within the astrotools module, whhich can be used as e.g.

```python
import ajrpy.astrotools as at

# Create an image header in Galactic coordinates based on a tuple with 6 items:
lcen = 23.9
bcen = 0.05
lsize = 3.0 # degrees
bsize = 2.0 # degrees
pixsize = 3.0 # arcsec
bunit = 'mJy/beam' # 'BUNIT' item for header gives pixel units

header = at.GalacticHeader((lcen, bcen, lsize, bsize, pixsize, bunit))
```

## Functions

`BeamArea` - Returns the area of a Gaussian beam with a given FWHM  
`ColBar` - Returns a simple matplotlib.pyplot.figure instance with some useful customization  
`CubeRMS` - Reurns an RMS map from a given cube by inverting the negative pixel values  
`GalacticHeader` - Create a new Galactic header, or convert an existing header to Galactic coordinates  
`reproject_Galactic` - Reprojects an equatorial .fits image onto a pixel grid in Galactic coordinates  
`RMS` - Return the RMS value or array for some input data  
`RoundUpToOdd` - Rounds a value up to the nearest odd integer  
