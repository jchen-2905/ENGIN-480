import numpy as np
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.recorders import TopFarmListRecorder
import topfarm
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014,Zong_PorteAgel_2020, Niayifar_PorteAgel_2016, CarbajoFuertes_etal_2018,Blondel_Cathelain_2020
from py_wake.utils.gradients import autograd
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site.shear import PowerShear
import pickle
with open('utm_boundary.pkl', 'rb') as f:
    boundary = np.array(pickle.load(f))
with open('utm_layout.pkl', 'rb') as f:
    xinit,yinit = np.array(pickle.load(f))

maxiter = 800
tol = 1e-6
class Vesta_V80_2(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name = 'Haliade-X 13 MW', diameter = 80, hub_height = 100, power_norm = 2000, turbulence_intensity=0.07)
class HornsRev1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.8837, 3.1401, 3.7864, 9.3933, 7.8709, 6.1294, 7.2347, 11.9692,  13.1866, 12.5732, 12.7453, 8.0871]
        a = [8.28, 8.97, 10.79, 11.45, 11.86, 11.20, 11.71, 12.98, 12.98, 12.11, 12.29, 10.53]
        k = [2.064, 3.072, 2.564, 2.861, 2.666, 2.467, 2.150, 2.432, 2.279, 2.283, 2.443, 2.010]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([xinit, yinit]).T
        self.name = "Horns Rev 1"

wind_turbines = Vesta_V80_2()
site = HornsRev1()
sim_res = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.0324555)
def aep_func(x,y):
    aep = sim_res(x,y).aep().sum()
    return aep

def daep_func(x,y):
    daep = sim_res.aep_gradients(gradient_method=autograd, wrt_arg=['x','y'], x=x,
    y=y)
    return daep

boundary_closed = np.vstack([boundary, boundary[0]])
cost_comp = CostModelComponent(input_keys=['x', 'y'],
n_wt = len(xinit),
cost_function = aep_func,
cost_gradient_function=daep_func,
objective=True,
maximize=True,
output_keys=[('AEP', 0)]
)
problem = TopFarmProblem(design_vars= {'x': xinit, 'y': yinit},
# constraints=[XYBoundaryConstraint(boundary),
constraints=[XYBoundaryConstraint(boundary, 'polygon'),
SpacingConstraint(334)],
cost_comp=cost_comp,
driver=EasyScipyOptimizeDriver(optimizer='SLSQP',
maxiter=maxiter, tol=tol),
n_wt=len(xinit),
expected_cost=0.001,
plot_comp=XYPlotComp()
)
cost, state, recorder = problem.optimize()
recorder.save('optimization_hr1')
print('done')
print('done')