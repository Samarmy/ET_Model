# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import xarray as xr
import ssebop
import  matplotlib.pyplot as plt
import math
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    band3 = xr.open_dataarray("D:\\PycharmProjects\\fineet-test\\landsat\\9x5_3_new.nc")
    band4 = xr.open_dataarray("D:\\PycharmProjects\\fineet-test\\landsat\\9x5_4_new.nc")
    band8 = xr.open_dataarray("D:\\PycharmProjects\\fineet-test\\landsat\\9x5_8_new.nc")
    date = "2020-01-16"

    ssebop_img = ssebop.Ssebop(
        band3,
        band4,
        band8,
        date,
        et_reference_factor=None,
        et_reference_resample="linear",
        tmax_resample="linear",
        elev_resample="linear",
    )

    print(ssebop_img.landsat_data)

    plt.clf()
    ssebop_img.interpolated_dt.plot()
    plt.savefig("interpolated_dt.png")

    plt.clf()
    ssebop_img.lst.plot()
    plt.savefig("lst.png")

    plt.clf()
    ssebop_img.tcorr.plot()
    plt.savefig("tcorr.png")

    plt.clf()
    ssebop_img.et_reference.plot()
    plt.savefig("et_reference.png")

    plt.clf()
    ssebop_img.et_fraction.plot()
    plt.savefig("et_fraction.png")

    plt.clf()
    ssebop_img.et.plot()
    plt.savefig("et.png")

    print(ssebop_img.error.values)
