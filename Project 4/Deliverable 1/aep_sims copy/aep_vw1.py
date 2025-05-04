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
class HaliadeX13MW(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name = 'Haliade-X 13 MW', diameter = 200, hub_height = 133, power_norm = 13000, turbulence_intensity=0.07)
class VineyardWind1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.4452, 7.6731, 6.4753, 6.0399, 4.8786, 4.5063, 7.3180, 11.7828, 13.0872,  11.1976, 11.1351, 9.4610]
        a = [10.26, 10.44, 9.52, 8.96, 9.58, 9.72, 11.48, 13.25, 12.46, 11.40, 12.35, 10.48]
        k = [2.225, 1.697, 1.721, 1.689, 1.525, 1.498, 1.686, 2.143, 2.369, 2.186, 2.385, 2.404]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([xinit, yinit]).T
        self.name = "Vineyard Wind 1"

wind_turbines = HaliadeX13MW()
site = VineyardWind1()
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
recorder.save('optimization_vw1')
print('done')
print('done')