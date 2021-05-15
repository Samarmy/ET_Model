import xarray as xr


def emissivity(landsat_image):
    """Emissivity as a function of NDVI

    Parameters
    ----------
    landsat_image : xarray
        "Prepped" Landsat image with standardized band names.

    Returns
    -------
    xarray

    References
    ----------

    """
    ndvi_img = ndvi(landsat_image)
    Pv = ((ndvi_img - 0.2) / 0.3) ** 2

    # ndviRangevalue = ndvi_image.where(
    #     ndvi_image.gte(0.2).And(ndvi_image.lte(0.5)), ndvi_image)
    # Pv = ndviRangevalue.expression(
    #     '(((ndviRangevalue - 0.2) / 0.3) ** 2',
    #     {'ndviRangevalue':ndviRangevalue})

    # Assuming typical Soil Emissivity of 0.97 and Veg Emissivity of 0.99
    #   and shape Factor mean value of 0.553
    dE = (1 - 0.97) * (1 - Pv) * (0.55 * 0.99)
    RangeEmiss = (0.99 * Pv) + (0.97 * (1 - Pv)) + dE

    # RangeEmiss = 0.989 # dE.expression(
    #  '((0.99 * Pv) + (0.97 * (1 - Pv)) + dE)', {'Pv':Pv, 'dE':dE})
    return ndvi_img \
        .where(ndvi_img.lt(0), 0.985) \
        .where(ndvi_img.gte(0) & (ndvi_img.lt(0.2)), 0.977) \
        .where(ndvi_img.gt(0.5), 0.99) \
        .where(ndvi_img.gte(0.2) & (ndvi_img.lte(0.5)), RangeEmiss) \
        .clip(0.977, 0.99)


def lst(landsat_image):
    """Emissivity corrected land surface temperature (LST) from brightness Ts.

    Parameters
    ----------
    landsat_image : xarray
        "Prepped" Landsat image with standardized band names.
        Image must also have 'k1_constant' and 'k2_constant' properties.

    Returns
    -------
    xarray

    Notes
    -----
    The corrected radiation coefficients were derived from a small number
    of scenes in southern Idaho [Allen2007] and may not be appropriate for
    other areas.

    References
    ----------


    Notes
    -----
    tnb = 0.866   # narrow band transmissivity of air
    rp = 0.91     # path radiance
    rsky = 1.32   # narrow band clear sky downward thermal radiation

    """
    # Get properties from image
    k1 = landsat_image["k1_constant"]  # K1 is for band10 = 774.89, band11 = 480.89
    k2 = landsat_image["k2_constant"]  # K2 is for band10 = 1321.08, band11 = 1201.14

    ts_brightness = landsat_image["tir"]

    # output_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir', 'BQA']
    # 'LANDSAT_8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'BQA'],

    emissivity_img = emissivity(landsat_image)

    # First back out radiance from brightness temperature
    # Then recalculate emissivity corrected Ts
    thermal_rad_toa = k1 / (xr.ufuncs.exp(k2 / ts_brightness) - 1)

    rc = ((thermal_rad_toa - 0.91) / 0.866) - ((1 - emissivity_img) * 1.32)

    lst = k2 / xr.ufuncs.log(emissivity_img * k1 / rc + 1)

    return lst


def ndvi(landsat_image):
    """Normalized difference vegetation index

    Parameters
    ----------
    landsat_image : xarray
        "Prepped" Landsat image with standardized band names.

    Returns
    -------
    xarray

    """
    # Note that negative input values are forced to 0 so that the result is confined to the range (-1, 1)
    # Hoping the values below are between (-1, 1) if not try something to make them get between (-1, 1)
    ndvi = ((landsat_image["nir"] - landsat_image["red"]) / (landsat_image["nir"] + landsat_image["red"]))
    return ndvi