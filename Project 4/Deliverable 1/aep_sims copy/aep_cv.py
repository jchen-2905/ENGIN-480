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
class SG_SWT_60_154(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SG SWT 6.0 154', diameter=154,
        hub_height=110,
        power_norm=6000, turbulence_intensity=0.08)
class Coastal_Virginia(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [9.1938, 9.9099, 9.0817, 5.2505, 4.8252, 5.7245, 11.4910, 14.2491, 9.3086, 5.0600, 6.4652, 9.4405]
        a = [10.50, 9.94, 8.96, 8.22, 7.34, 7.94, 11.27, 13.33, 11.86, 10.03, 10.26, 11.12]
        k = [2.260, 2.139, 1.971, 1.771, 1.521, 1.514, 1.955, 2.568, 2.775, 2.049, 1.951, 2.295]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti,
        shear=shear)
        self.initial_position = np.array([xinit, yinit]).T
        self.name = "Coastal Virginia"
wind_turbines = SG_SWT_60_154()

site = Coastal_Virginia()
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
plot_comp=XYPlotComp(),
recorder=TopFarmListRecorder(None, ['AEP', 'x', 'y'])
)
cost, state, recorder = problem.optimize()
recorder.save('optimization_cv')

print('done')
