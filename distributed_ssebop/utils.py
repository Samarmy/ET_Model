import xarray as xr
import requests
from os import mkdir
from os.path import isdir


def get_nc_files(year):
    if not isdir("distributed_ssebop\\data"):
        mkdir("distributed_ssebop\\data")

    if not isdir("distributed_ssebop\\data\\gridmet"):
        mkdir("distributed_ssebop\\data\\gridmet")

    mkdir("distributed_ssebop\\data\\gridmet\\" + year)

    url = 'https://www.northwestknowledge.net/metdata/data/tmmn_' + year + '.nc'
    r = requests.get(url)
    with open('distributed_ssebop\\data\\gridmet\\' + year + '\\tmin.nc', 'wb') as f:
        f.write(r.content)

    url = 'https://www.northwestknowledge.net/metdata/data/tmmx_' + year + '.nc'
    r = requests.get(url)
    with open('distributed_ssebop\\data\\gridmet\\' + year + '\\tmax.nc', 'wb') as f:
        f.write(r.content)

    url = 'https://www.northwestknowledge.net/metdata/data/etr_' + year + '.nc'
    r = requests.get(url)
    with open('distributed_ssebop\\data\\gridmet\\' + year + '\\etr.nc', 'wb') as f:
        f.write(r.content)


def rmse(predicted, actual):
    mse = xr.ufuncs.square(actual.subtract(predicted)).mean()
    rmse = xr.ufuncs.sqrt(mse)
    return rmse
