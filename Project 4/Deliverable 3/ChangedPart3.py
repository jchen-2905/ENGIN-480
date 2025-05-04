import matplotlib
matplotlib.use('TkAgg')
import matplotlib as plt

import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath

examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
examples_sites

name = 'France_good_wind'
ex_site = examples_sites.loc[examples_sites.name == name]

longitude = ex_site['longitude'].values[0]
latitude = ex_site['latitude'].values[0]
altitude = ex_site['altitude'].values[0]

input_ts_fn = examples_filepath + ex_site['input_ts_fn'].values[0]

input_ts = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)

required_cols = [col for col in input_ts.columns if 'WD' not in col]
input_ts = input_ts.loc[:, required_cols]
input_ts

sim_pars_fn = examples_filepath + ex_site['sim_pars_fn'].values[0]

with open(sim_pars_fn) as file:
    sim_pars = yaml.load(file, Loader=yaml.FullLoader)

print(sim_pars_fn)
sim_pars

rotor_diameter_m = 100     # was 220
hub_height_m     = 80      # was 130
wt_rated_power_MW = 10
surface_tilt_deg = 35
surface_azimuth_deg = 180
DC_AC_ratio = 1.5

hpp = hpp_model(
    latitude=latitude,
    longitude=longitude,
    altitude=altitude,
    rotor_diameter_m=rotor_diameter_m,
    hub_height_m=hub_height_m,
    wt_rated_power_MW=wt_rated_power_MW,
    surface_tilt_deg=surface_tilt_deg,
    surface_azimuth_deg=surface_azimuth_deg,
    DC_AC_ratio=DC_AC_ratio,
    num_batteries=5,
    work_dir='./',
    sim_pars_fn=sim_pars_fn,
    input_ts_fn=input_ts_fn,
)

# price_fn = examples_filepath + ex_site['price_fn'].values[0]
# price = pd.read_csv(price_fn, index_col=0, parse_dates=True)[ex_site.price_col.values[0]]

# hpp = hpp_model_simple(
#     latitude,
#     longitude,
#     altitude,
#     rotor_diameter_m=rotor_diameter_m,
#     hub_height_m=hub_height_m,
#     wt_rated_power_MW=wt_rated_power_MW,
#     surface_tilt_deg=surface_tilt_deg,
#     surface_azimuth_deg=surface_azimuth_deg,
#     DC_AC_ratio=DC_AC_ratio,
#     num_batteries=1,
#     work_dir='./',
#     sim_pars_fn=sim_pars_fn,
#     input_ts_fn=None,
#     price_fn=price,
# )

start = time.time()

Nwt = 20
wind_MW_per_km2 = 7
solar_MW = 150
b_P = 20
b_E_h = 3
cost_of_batt_degr = 5
clearance = hub_height_m - rotor_diameter_m / 2
sp = 4 * wt_rated_power_MW * 10 ** 6 / np.pi / rotor_diameter_m ** 2

x = [
    # Wind plant design
    clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
    # PV plant design
    solar_MW, surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
    # Energy storage & EMS price constrains
    b_P, b_E_h, cost_of_batt_degr
]

outs = hpp.evaluate(*x)

hpp.print_design(x, outs)

end = time.time()
print(f'exec. time [min]:', (end - start) / 60)

b_E_SOC_t = hpp.prob.get_val('ems.b_E_SOC_t')
b_t = hpp.prob.get_val('ems.b_t')
price_t = hpp.prob.get_val('ems.price_t')

wind_t = hpp.prob.get_val('ems.wind_t')
solar_t = hpp.prob.get_val('ems.solar_t')
hpp_t = hpp.prob.get_val('ems.hpp_t')
hpp_curt_t = hpp.prob.get_val('ems.hpp_curt_t')
grid_MW = hpp.prob.get_val('ems.G_MW')

n_days_plot = 14
plt.figure(figsize=[12, 4])
plt.plot(price_t[:24 * n_days_plot], label='price')
plt.plot(b_E_SOC_t[:24 * n_days_plot], label='SoC [MWh]')
plt.plot(b_t[:24 * n_days_plot], label='Battery P [MW]')
plt.xlabel('time [hours]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=3, fancybox=0, shadow=0)

plt.figure(figsize=[12, 4])
plt.plot(wind_t[:24 * n_days_plot], label='wind')
plt.plot(solar_t[:24 * n_days_plot], label='PV')
plt.plot(hpp_t[:24 * n_days_plot], label='HPP')
plt.plot(hpp_curt_t[:24 * n_days_plot], label='HPP curtailed')
plt.axhline(grid_MW, label='Grid MW', color='k')
plt.xlabel('time [hours]')
plt.ylabel('Power [MW]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=5, fancybox=0, shadow=0)

N_life = sim_pars['N_life']
life_h = N_life * 365 * 24
age = np.arange(life_h) / (24 * 365)

SoH = hpp.prob.get_val('battery_degradation.SoH')
SoH_all = np.copy(hpp.prob.get_val('battery_loss_in_capacity_due_to_temp.SoH_all'))

wind_t_ext = hpp.prob.get_val('ems_long_term_operation.wind_t_ext')
wind_t_ext_deg = hpp.prob.get_val('ems_long_term_operation.wind_t_ext_deg')

solar_t_ext = hpp.prob.get_val('ems_long_term_operation.solar_t_ext')
solar_t_ext_deg = hpp.prob.get_val('ems_long_term_operation.solar_t_ext_deg')

hpp_t = hpp.prob.get_val('ems.hpp_t')
hpp_t_with_deg = hpp.prob.get_val('ems_long_term_operation.hpp_t_with_deg')

df = pd.DataFrame(
    index=pd.date_range(start='2023-01-01', end='2048-01-01', freq='1h'),
)
df['wind_t_ext'] = np.nan
df['wind_t_ext_deg'] = np.nan
df['solar_t_ext'] = np.nan
df['solar_t_ext_deg'] = np.nan
df['hpp_t'] = np.nan
df['hpp_t_with_deg'] = np.nan

df.iloc[:len(age):, 0] = wind_t_ext
df.iloc[:len(age):, 1] = wind_t_ext_deg
df.iloc[:len(age):, 2] = solar_t_ext
df.iloc[:len(age):, 3] = solar_t_ext_deg
df.iloc[:len(age):, 4] = hpp_t
df.iloc[:len(age):, 5] = hpp_t_with_deg

df = df.dropna(axis=0)

df_year = df.groupby(df.index.year).mean()
df_year['age'] = np.arange(len(df_year)) + 0.5

df_year['eff_wind_ts_deg'] = df_year.wind_t_ext_deg.values / df_year.wind_t_ext.values
df_year['eff_solar_ts_deg'] = df_year.solar_t_ext_deg.values / df_year.solar_t_ext.values
df_year['eff_hpp_ts_deg'] = df_year.hpp_t_with_deg.values / df_year.hpp_t.values

plt.figure(figsize=[12, 4])
plt.plot(df_year.age.values, df_year.eff_wind_ts_deg.values, label='Wind degr.')
plt.plot(df_year.age.values, df_year.eff_solar_ts_deg.values, label='Solar degr.')
plt.plot(df_year.age.values, df_year.eff_hpp_ts_deg.values, '--', label='HPP degr.')
plt.plot(age, SoH, label='Battery degr.')
plt.plot(age, SoH_all, label='Battery degr. and low temp. losses', alpha=0.5)

plt.legend()

plt.xlabel('age [years]')
plt.ylabel('CF_deg/CF for wind, solar and hpp [-] \n Battery loss of storing capacity [-]')

cost_of_battery_P_fluct_in_peak_price_ratio = 0.0
x = [
    clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
    solar_MW, surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
    b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio
]
outs = hpp.evaluate(*x)

SoH = np.copy(hpp.prob.get_val('battery_degradation.SoH'))

cost_of_battery_P_fluct_in_peak_price_ratio_B = 5
x = [
    clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
    solar_MW, surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
    b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio_B
]
outs = hpp.evaluate(*x)
SoH_B = np.copy(hpp.prob.get_val('battery_degradation.SoH'))

cost_of_battery_P_fluct_in_peak_price_ratio_C = 20
x = [
    clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
    solar_MW, surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
    b_P, b_E_h, cost_of_batt_degr
]
outs = hpp.evaluate(*x)
SoH_C = np.copy(hpp.prob.get_val('battery_degradation.SoH'))

plt.figure(figsize=[12, 3])
plt.plot(age, SoH, label=r'$C_{bfl}=0$')
# plt.plot(age, SoH_B, label=f'{cost_of_battery_P_fluct_in_peak_price_ratio_B}*Pr_Peak')
plt.plot(age, SoH_C, label=r'$C_{bfl}=$' + f'{cost_of_battery_P_fluct_in_peak_price_ratio_C}')
plt.plot(age, 0.7 * np.ones_like(age), label=r'$min(1-L) = 0.7$', color='r', alpha=0.5)
plt.xlabel('age [years]')
plt.ylabel(r'Battery State of Health, $1-L(t)$ [-]')
plt.legend(
    title='Cost of Battery fluctuations',
    loc='upper center', bbox_to_anchor=(0.5, 1.27),
    ncol=3, fancybox=0, shadow=0
)

plt.show()
