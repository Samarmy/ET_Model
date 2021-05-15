import xarray as xr
import numpy as np


def interpolate(data, data_name, reference, type_of_interpolation):
    if type_of_interpolation in ["linear"]:
        return linear_interpolation(data, data_name, reference)
    else:
        raise ValueError('unsupported interpolation: bilinear')


def linear_interpolation(data, data_name, reference):
    if data_name == "":
        interpolated_data = reference.assign(
            interpolate=lambda each_data: data.interp(lat=reference.lat,
                                                      lon=reference.lon))
    else:
        interpolated_data = reference.assign(
            interpolate=lambda each_data: data[data_name].interp(lat=reference.lat,
                                                                 lon=reference.lon))
    interpolated_data = xr.DataArray(
        data=np.flipud(interpolated_data.interpolate.values),
        dims=["y", "x"],
        coords=dict(
            lon=(["y", "x"], interpolated_data.lon),
            lat=(["y", "x"], interpolated_data.lat),
        ),
    )
    return interpolated_data


def bilinear_interpolation(data, data_name, reference):
    raise ValueError('unsupported interpolation: bilinear')
