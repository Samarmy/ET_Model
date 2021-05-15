from distributed_ssebop import utils
import xarray as xr
import math
import numpy as np

from distributed_ssebop import landsat
from distributed_ssebop import model
from distributed_ssebop.interpolate import interpolate

from os.path import isfile, join, isdir


class Ssebop():
    """Xarray based SSEBop Image"""

    def __init__(
            self, band3_data,
            band4_data,
            band8_data,
            date,
            et_reference_factor=None,
            et_reference_resample="linear",
            tmax_resample="linear",
            elev_resample="linear",
            dt_resample="linear",
    ):
        self.band3_data = band3_data
        self.band4_data = band4_data
        self.band8_data = band8_data
        self.date = date

        date_split = date.split("-")
        year = date_split[0]

        gridmet_path = "distributed_ssebop\\data\\gridmet\\"

        if not isdir(gridmet_path + year):
            utils.get_nc_files(year)

        tmin_data = xr.open_dataset("distributed_ssebop\\data\\gridmet\\" + year + "\\tmin.nc").sel(day=date)
        tmax_data = xr.open_dataset("distributed_ssebop\\data\\gridmet\\" + year + "\\tmax.nc").sel(day=date)
        etr_data = xr.open_dataset("distributed_ssebop\\data\\gridmet\\" + year + "\\etr.nc").sel(day=date)

        tmin_data["tmin"] = tmin_data["air_temperature"]
        tmin_data = tmin_data.drop(["air_temperature"])
        tmin_data = tmin_data.drop_dims("crs")

        tmax_data["tmax"] = tmax_data["air_temperature"]
        tmax_data = tmax_data.drop(["air_temperature"])
        tmax_data = tmax_data.drop_dims("crs")

        etr_data["etr"] = etr_data["potential_evapotranspiration"]
        etr_data = etr_data.drop(["potential_evapotranspiration"])
        etr_data = etr_data.drop_dims("crs")

        self.tmin_data = tmin_data
        self.tmax_data = tmax_data
        self.et_reference_data = etr_data

        self.elev_data = xr.open_dataset("distributed_ssebop\\data\\gridmet\\" + "elev.nc").sel(day=1)

        self.et_reference_factor = et_reference_factor
        self.et_reference_resample = et_reference_resample
        self.tmax_resample = tmax_resample
        self.elev_resample = elev_resample
        self.dt_resample = dt_resample

        self.landsat_data = self.band_to_landsat()
        self.dt = self.dt()
        self.interpolated_dt = self.interpolated_dt()
        self.lst = self.lst()
        self.interpolated_tmax = self.interpolated_tmax()
        self.tcorr = self.tcorr()
        self.interpolated_elev = self.interpolated_elev()
        self.et_reference = self.et_reference()
        self.et_fraction = self.et_fraction()
        self.et = self.et()
        self.error = self.error()

    def et_fraction(self):
        """Fraction of reference ET"""
        et_fraction = model.et_fraction(
            lst=self.lst, tmax=self.interpolated_tmax, tcorr=self.tcorr,
            dt=self.interpolated_dt, elev=self.interpolated_elev, elr_flag=False,
        )

        return et_fraction

    def et_reference(self):
        """Reference ET for the date"""
        if self.et_reference_resample in ['linear']:
            et_reference = interpolate(self.et_reference_data, "etr",
                                       self.landsat_data,
                                       self.et_reference_resample)

        if self.et_reference_factor:
            et_reference = et_reference.multiply(self.et_reference_factor)

        return et_reference

    def et(self):
        return self.et_fraction.multiply(self.et_reference)

    def dt(self):
        return model.dt(
            tmax=self.tmax_data["tmax"],
            tmin=self.tmin_data["tmin"],
            elev=self.elev_data["elev"],
            lat=self.tmax_data["lat"],
            doy=None,
            rs=None,
            ea=None
        )

    def interpolated_tmax(self):
        if self.tmax_resample in ['linear']:
            tmax = interpolate(self.tmax_data, "tmax", self.landsat_data,
                               self.tmax_resample)
        return tmax

    def interpolated_elev(self):
        if self.elev_resample in ['linear']:
            elev = interpolate(self.elev_data, "elev", self.landsat_data,
                               self.elev_resample)
        return elev

    def interpolated_dt(self):
        if self.dt_resample in ['linear']:
            data = interpolate(self.dt, "", self.landsat_data,
                               self.dt_resample)
        return data

    def lst(self):
        return landsat.lst(self.landsat_data)

    def tcorr(self):
        return self.lst.divide(self.interpolated_tmax)

    def band_to_landsat(self):
        band3 = self.band3_data.multiply(0.0001)
        band4 = self.band4_data.multiply(0.0001)
        band8 = self.band8_data.multiply(0.1)

        land_sat_data = xr.Dataset(
            data_vars=dict(
                red=(["y", "x"], band3),
                nir=(["y", "x"], band4),
                tir=(["y", "x"], band8),
                k1_constant=(774.89),
                k2_constant=(1321.08)
            ),
            coords=dict(
                lon=(["y", "x"], band3.lon.values),
                lat=(["y", "x"], band3.lat.values),
            ),
        )

        return land_sat_data

    def error(self):
        return utils.rmse(self.et, self.et_reference)
