import numpy as np
import xarray as xr
from py_wake.site import XRSite

def build_xrsite_from_lib(data_dict):
    heights = sorted(data_dict.keys())             
    num_dirs = len(next(iter(data_dict.values()))) 
    wd = np.arange(0, 360, 360 // num_dirs)

    # Initialize arrays with correct shape: [n_dirs, n_heights]
    A_arr = np.zeros((num_dirs, len(heights)))
    k_arr = np.zeros_like(A_arr)
    f_arr = np.zeros((num_dirs,)) 

    for h_idx, h in enumerate(heights):
        for wd_idx, (f, A, k) in enumerate(data_dict[h]):
            A_arr[wd_idx, h_idx] = A
            k_arr[wd_idx, h_idx] = k
            f_arr[wd_idx] = f 

    # Normalize frequencies
    f_arr /= f_arr.sum()

    # Constant turbulence intensity
    TI_arr = 0.1 * np.ones_like(A_arr)

    # Create Dataset for XRSite
    ds = xr.Dataset(
        data_vars={
            'Sector_frequency': ('wd', f_arr),
            'Weibull_A': (('wd', 'h'), A_arr),
            'Weibull_k': (('wd', 'h'), k_arr),
            'TI': (('wd', 'h'), TI_arr)
        },
        coords={
            'wd': wd,
            'h': heights
        }
    )

    return XRSite(ds)


from gwa_parser import gwa3_data_parser

data = gwa3_data_parser("gwa3_HornsRev1_gwc.lib")
site = build_xrsite_from_lib(data)

print("XRSite created:", site)


