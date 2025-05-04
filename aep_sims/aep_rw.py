import numpy as np
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
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
class SG_110_200_DD(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SG 11.0-200 DD', diameter=200,
        hub_height=140,
        power_norm=11000, turbulence_intensity=0.08)
class Revolutionwind_southforkwind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [7.2913, 7.2204, 6.3564, 5.5052, 4.743, 4.7018, 7.7244, 11.6506,
        13.331, 11.079, 10.9413, 9.4554]
        a = [10.37, 10.58, 9.66, 9.33, 9.68, 10.57, 11.77, 13.87, 12.79, 12.12,
        12.36, 10.3]
        k = [2.053, 1.729, 1.635, 1.689, 1.412, 1.42, 1.529, 1.943, 2.076, 2.197,
        2.295, 2.201]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti,
        shear=shear)
        self.initial_position = np.array([xinit, yinit]).T
        self.name = "Revolutionwind Southforkwind"
wind_turbines = SG_110_200_DD()

site = Revolutionwind_southforkwind()
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
recorder.save('optimization_rw')
print('done')
print('done')