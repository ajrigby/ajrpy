# ajrpy
A module containing generally useful functions for astronomy and data analysis

## Installation
The simplest way to install this package is through pip:

```pip install git+https://github.com/ajrigby/ajrpy.git```

## Usage
Most of the useful functions are located within the astrotools module, whhich can be used as e.g.

```
import ajrpy.astrotools as at

header = at.GalacticHeader((23.9, 0.05, 3.0, 2.0, 3.0, 'mJy/beam'))
```

## Functions

`GalacticHeader` - Create a new Galactic header, or convert an existing header to Galactic coordinates
