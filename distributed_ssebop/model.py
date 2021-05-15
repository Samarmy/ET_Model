import xarray as xr
import math


def et_fraction(lst, tmax, tcorr, dt, elev=None, elr_flag=True):
    """SSEBop fraction of reference ET (ETf)

    Parameters
    ----------
    lst : xarray
        Land surface temperature (lst) [L].
    tmax : xarray
        Maximum air temperature [K].
    tcorr : xarray
        Tcorr.
    dt : xarray
        Temperature difference [K].
    elr_flag : bool, optional
        If True, apply Elevation Lapse Rate (ELR) adjustment
        (the default is False).
    elev : xarray
        Elevation [m] (the default is None).  Only needed if elr_flag is True.

    Returns
    -------
    xarray

    References
    ----------


    """
    # Adjust air temperature based on elevation (Elevation Lapse Rate)
    if elr_flag:
        tmax = lapse_adjust(tmax, elev)

    #     et_fraction = ((lst * (-1) + tmax * tcorr +dt)/lst)
    et_fraction = ((lst * (-1) + tmax * tcorr + dt) / dt)
    #     et_fraction_temp = et_fraction
    #     et_fraction = et_fraction.where(et_fraction<0, 0).where(et_fraction>1.05, 1.05)
    # et_fraction \
    #         .updateMask(et_fraction.lt(1.5))\
    #         .clamp(0, 1.05)

    return et_fraction


def dt(tmax, tmin, elev, lat, doy=None, rs=None, ea=None):
    """Temperature difference between hot/dry ground and cold/wet canopy

    Parameters
    ----------
    tmax : xarray
        Maximum daily air temperature [K].
    tmin : xarray
        Maximum daily air temperature [K].
    elev : xarray
        Elevation [m].
    doy : xarray
        Day of year.
    lat : xarray
        Latitude [deg].  If not set, use GEE pixelLonLat() method.
    rs : xarray
        Incoming solar radiation [MJ m-2 d-1].  If not set the theoretical
        clear sky solar (Rso) will be used for the Rs.
    ea : xarray
        Actual vapor pressure [kPa].  If not set, vapor pressure will be
        computed from Tmin.

    Returns
    -------
    xarray

    Raises
    ------
    ValueError if doy is not set.

    References
    ----------

    """

    if doy is None:
        doy = tmax.multiply(0).add(1)

    # Convert latitude to radians
    phi = lat.multiply(math.pi / 180)

    # Make a DOY image from the DOY number
    doy = tmax.multiply(0).add(doy)

    # Extraterrestrial radiation (Ra) (FAO56 Eqns 24, 25, 23, 21)
    delta = doy.multiply(2 * math.pi / 365).subtract(1.39).sin() \
        .multiply(0.409)
    ws = phi.tan().multiply(-1).multiply(delta.tan()).acos()
    dr = doy.multiply(2 * math.pi / 365).cos().multiply(0.033).add(1)
    ra = ws.multiply(phi.sin()).multiply(delta.sin()) \
        .add(phi.cos().multiply(delta.cos()).multiply(ws.sin())) \
        .multiply(dr) \
        .multiply((1367.0 / math.pi) * 0.0820)

    # Simplified clear sky solar formulation (Rso) [MJ m-2 d-1] (Eqn 37)
    rso = elev.multiply(2E-5).add(0.75).multiply(ra)

    # Derive cloudiness fraction from Rs and Rso (see FAO56 Eqn 39)
    # Use Rso for Rs if not set
    if rs is None:
        rs = rso.multiply(1)
        fcd = 1
    else:
        fcd = rs.divide(rso).max(0.3).min(1.0).multiply(1.35).subtract(0.35)
        # fcd = rs.divide(rso).clamp(0.3, 1).multiply(1.35).subtract(0.35)

    # Net shortwave radiation [MJ m-2 d-1] (FAO56 Eqn 38)
    rns = rs.multiply(1 - 0.23)

    # Actual vapor pressure [kPa] (FAO56 Eqn 14)
    if ea is None:
        ea = tmin.subtract(273.15).multiply(17.27) \
            .divide(tmin.subtract(273.15).add(237.3)).exp().multiply(0.6108)

    # Net longwave radiation [MJ m-2 d-1] (FAO56 Eqn 39)
    rnl = tmax.pow(4).add(tmin.pow(4)) \
        .multiply(ea.sqrt().multiply(-0.14).add(0.34)) \
        .multiply(4.901E-9 * 0.5).multiply(fcd)

    # Net radiation [MJ m-2 d-1] (FAO56 Eqn 40)
    rn = rns.subtract(rnl)

    # Air pressure [kPa] (FAO56 Eqn 7)
    pair = elev.multiply(-0.0065).add(293.0).divide(293.0).pow(5.26) \
        .multiply(101.3)

    # Air density [Kg m-3] (Senay2018 A.11 & A.13)
    den = tmax.add(tmin).multiply(0.5).pow(-1).multiply(pair) \
        .multiply(3.486 / 1.01)

    # Temperature difference [K] (Senay2018 A.5)
    dt = rn.divide(den).multiply(110.0 / ((1.013 / 1000) * 86400))

    return dt


def lapse_adjust(temperature, elev, lapse_threshold=1500):
    """Elevation Lapse Rate (ELR) adjusted temperature [K]

    Parameters
    ----------
    temperature : xarray
        Temperature [K].
    elev : xarray
        Elevation [m].
    lapse_threshold : float
        Minimum elevation to adjust temperature [m] (the default is 1500).

    Returns
    -------
    xarray

    """
    new_temperature = (temperature - (0.003 * (elev - lapse_threshold)))
    return temperature.where(elev.gt(lapse_threshold), new_temperature)
