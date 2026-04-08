import numpy as np
import dedalus.public as d3
from scipy import interpolate
from dedalus.core.domain import Domain
from .utils.read_left_bc import leftbc

def gaussian_bathymetry(x: np.ndarray, params: tuple[float, float]) -> np.ndarray:
    """Gaussian bathymetry function.
    Args:
        x: The x-coordinate.

    Returns:
        The bathymetry value at x.
    """
    return 0.2*np.exp(-1/params[1]*(x-params[0])**2)

def rampFunc(x: np.ndarray) -> np.ndarray:
    """Exact bathymetry function from waterchannel experiment.
    Args:
        x: The x-coordinate.

    Returns:
        The bathymetry value at x."""
    b_points = np.concatenate((np.zeros(4),
                                    np.array([0, 0.024, 0.053, 0.0905, 0.133, 0.182, 0.2, 0.182, 0.133,
                                    0.0905, 0.053, 0.024, 0]),
                                    np.zeros(21)))
    x_points = np.concatenate((np.arange(1.5, 3.5, 0.5),
                                np.array([3.4125, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.5875]),
                                np.arange(5, 15.5, 0.5)))
    itp_bathy = interpolate.PchipInterpolator(x_points, b_points)
    return itp_bathy(x)


class CustomDomain:
    """Domain for the shallow water equations.

    Attributes:
        xcoord: The x-coordinate.
        dist: The distributor.
        xbasis: The x-basis.
        dom: The domain.
        x: The x-grid.
    """
    def __init__(self, xbound: tuple[float, float], nx: int, dealias: float,
                 basis_type=d3.RealFourier):
        self.xcoord = d3.Coordinate('x')
        self.dist = d3.Distributor(self.xcoord, dtype=np.float64)
        self.xbasis = basis_type(self.xcoord, size=nx, bounds=xbound,
                         dealias=dealias)
        self.dom = Domain(self.dist, bases=[self.xbasis])
        self.x = self.dist.local_grid(self.xbasis)

class PeriodicDomain(CustomDomain):
    """Domain for the periodic problem."""
    def __init__(self, xbound: tuple[float, float], nx: int, dealias: float):
        super().__init__(xbound, nx, dealias, basis_type=d3.RealFourier)

class WaterChannelDomain(CustomDomain):
    """Domain for the water channel problem."""
    def __init__(self, xbound: tuple[float, float], nx: int, dealias: float):
        super().__init__(xbound, nx, dealias, basis_type=d3.Chebyshev)


class InitialConditions:
    """Initial conditions for the shallow water equations.

    Attributes:
        h: The water height.
        u: The velocity.
        t: The time.
        b: The bathymetry.
        H: The total water height.
    """
    def __init__(self, domain: CustomDomain):
        self.h = domain.dist.Field(name='h', bases=domain.xbasis)
        self.u = domain.dist.Field(name='u', bases=domain.xbasis)
        self.t = domain.dist.Field()
        self.b = domain.dist.Field(bases=domain.xbasis)
        self.H = 0.3

    def change_space_scales(self, scale: float):
        """Change the space scales of the fields.
        Args:
            scale: The scale factor.
        """
        self.h.change_scales(scale)
        self.u.change_scales(scale)
        self.b.change_scales(scale)

    def set_water_level(self, level: float):
        """Set the water level.
        Args:
            level: The water level.
        """
        self.H = level


class PeriodicInitialConditions(InitialConditions):
    """Initial conditions for the periodic problem."""
    def __init__(self, domain: CustomDomain, xbound: tuple[float, float]=(0., 10.)):
        super().__init__(domain)
        width = xbound[1] - xbound[0]
        self.H = 0.3 + 0.5 * np.exp(-(domain.x - (width/2+xbound[0])) ** 2 / 2 ** 2)\
                * 0.05 * np.sin(2 * np.pi/width * (domain.x- xbound[0]))


class WaterChannelInitialConditions(InitialConditions):
    """Initial conditions for the water channel problem."""
    def __init__(self, domain: CustomDomain, tstart: float=32.):
        super().__init__(domain)
        self.bc_field = domain.dist.Field()
        self.tau1 = domain.dist.Field(name='tau1')
        self.tau2 = domain.dist.Field(name='tau2')
        self.tstart = tstart

    def change_space_scales(self, scale: float):
        """Change the space scales of the fields.
        Args:
            scale: The scale factor.
        """
        self.h.change_scales(scale)
        self.u.change_scales(scale)
        self.b.change_scales(scale)
        self.bc_field.change_scales(scale)
        self.tau1.change_scales(scale)
        self.tau2.change_scales(scale)


class Solver:
    """Base solver for the shallow water equations.

    Attributes:
        dx: The x-derivative.
        name_dict: The dictionary of names.
        ic: The initial conditions.
        domain: The domain.
    """
    def __init__(self, domain: CustomDomain, initial_conditions: InitialConditions, params: dict):
        self.dx = lambda A: d3.Differentiate(A, domain.xcoord)

        self.name_dict = {
            'h': initial_conditions.h,
            'u': initial_conditions.u,
            't': initial_conditions.t,
            'b': initial_conditions.b,
            'g': params['g'],
            'kappa': params['kappa'],
            'dx': self.dx
        }

        self.ic = initial_conditions
        self.domain = domain

    def get_problem(self):
        """Get the solver for the problem.

        Returns:
            The solver.
        """
        raise NotImplementedError("This method should be implemented by subclasses")


class PeriodicSolver(Solver):
    """Solver for the periodic problem."""

    def get_problem(self):
        """Periodic boundary condition solver."""
        problem = d3.IVP([self.ic.h, self.ic.u],
                        time=self.ic.t, namespace=self.name_dict)
        problem.add_equation("dt(h) = -dx(h*u)")
        problem.add_equation("dt(u) + g*dx(h) + kappa*u = - g*dx(b) - u*dx(u)")

        solver = problem.build_solver(d3.RK443)
        return solver


class WaterChannelSolver(Solver):
    """Solver for the water channel problem."""

    def __init__(self, domain: WaterChannelDomain,
                 initial_conditions: WaterChannelInitialConditions,
                 params: dict, bcfile: str='./test_data/mean_bc.txt'):
        super().__init__(domain, initial_conditions, params)
        self.lbc = leftbc(bcfile)

    def get_problem(self):
        """Water channel boundary condition solver."""
        def hl_function(*args):
            t = args[0].data
            htemp = self.lbc.f(t + self.ic.tstart)
            self.ic.bc_field["g"] = self.ic.H + htemp - self.ic.b['g'][0]
            return self.ic.bc_field["g"]

        def hl(*args, domain=self.ic.bc_field.domain, F=hl_function):
            return d3.GeneralFunction(self.domain.dist, domain, layout='g', tensorsig=(),
                                      dtype=np.float64, func=F, args=args)

        lift_basis = self.domain.xbasis.derivative_basis(1)
        lift = lambda A, n: d3.Lift(A, lift_basis, n)
        tau1 = self.domain.dist.Field(name='tau1')
        tau2 = self.domain.dist.Field(name='tau2')

        self.name_dict['lift'] = lift
        self.name_dict['tau1'] = tau1
        self.name_dict['tau2'] = tau2
        self.name_dict['hl'] = hl

        # Problem
        problem = d3.IVP([self.ic.h, self.ic.u, tau1, tau2], time=self.ic.t,
                         namespace=self.name_dict)
        problem.add_equation("dt(h) + lift(tau1, -1) + dx(u) = - dx((h-1)*u)")
        problem.add_equation("dt(u) + lift(tau2, -1) + g*dx(h) + kappa*u = - u*dx(u) - g*dx(b)")
        problem.add_equation("h(x='left') = hl(t)")
        problem.add_equation("u(x='right') = 0")

        # Build solver
        solver = problem.build_solver(d3.RK443)
        return solver


class SWESolver():
    """Shallow water equation solver.

    Attributes:
        xbound: The x-boundary.
        dt: The time step.
        nx: The number of grid points.
        total_t: The end time.
        g: The gravity constant.
        kappa: The bottom friction coefficient.
        dealias: The dealiasing factor.

    Methods:
        __init__: Initialize the SWESolver.
        solve: Solve the shallow water equations.
    """

    def __init__(self, xbound: tuple[float, float], dt: float, nx: int, total_t: float, tstart: float=32.,
                 g: float=9.81, kappa: float=0.2, dealias: float=2/3, problemtype: str='periodic', bc_file: str=''):

        self.xbound = xbound
        self.dt = dt
        self.nx = nx
        self.total_t = total_t
        self.params = {'g': g, 'kappa': kappa, 'dealias': dealias}
        self.problemtype = problemtype

        # Initialize domain, initial conditions, and solver
        if self.problemtype == 'periodic':
            self.domain = PeriodicDomain(self.xbound, self.nx, self.params['dealias'])
            self.ic= PeriodicInitialConditions(self.domain, self.xbound)
            self.solver = PeriodicSolver(self.domain, self.ic, self.params)
        elif self.problemtype == 'waterchannel':
            self.domain = WaterChannelDomain(self.xbound, self.nx, self.params['dealias'])
            self.ic = WaterChannelInitialConditions(self.domain, tstart=tstart)
            self.solver = WaterChannelSolver(self.domain, self.ic, self.params, bcfile=bc_file)
        else:
            raise ValueError("Invalid problem type")


    def solve(self, b_array: np.ndarray, sensor_pos:np.ndarray = np.array([3.5, 5.5, 7.5])):
        """Solve the shallow water equations.
        Args:
            peak: The peak of the Gaussian bathymetry.

        Returns:
            H_sensor: The water height at the sensor positions.
            t_list: The array of time.
            h_list: The array of water height.
            u_list: The array of velocity.
        """

        # Set the initial conditions
        self.ic.b['g'] = np.squeeze(b_array)
        self.ic.h['g'] = self.ic.H - self.ic.b['g']
        self.ic.u['g'] = 0.0

        self.ic.tau1['g'] = 0.0
        self.ic.tau2['g'] = 0.0
        self.ic.bc_field['g'] = 0.0
        self.ic.t['g'] = 0.0

        b_sensor = self.eval_at_sensor_positions(self.ic.b, sensor_pos)
        temph_sensor = self.eval_at_sensor_positions(self.ic.h, sensor_pos) + b_sensor
        self.ic.change_space_scales(1)
        # Set parameters for the solver

        solver = self.solver.get_problem()
        solver.sim_time = 0.0
        solver.iteration = 0

        solver.stop_iteration = int(self.total_t/abs(self.dt))
        solver.stop_sim_time = self.total_t #- 1e-13

        H_sensor_list = [temph_sensor]
        h_complete_list = [np.copy(self.ic.h['g'])]
        u_complete_list = [np.copy(self.ic.u['g'])]
        t_list = [solver.sim_time]

        while solver.proceed:
            solver.step(self.dt)
            temph_sensor = self.eval_at_sensor_positions(self.ic.h, sensor_pos) + b_sensor
            H_sensor_list.append(temph_sensor)
            self.ic.h.change_scales(1)
            h_complete_list.append(np.copy(self.ic.h['g']))
            self.ic.u.change_scales(1)
            u_complete_list.append(np.copy(self.ic.u['g']))
            t_list.append(solver.sim_time)
            if np.max(self.ic.h['g']) > 100:
                break

        self.ic.change_space_scales(1)
        H_sensor = np.array(H_sensor_list)
        return H_sensor,  np.array(t_list),\
                np.array(h_complete_list), np.array(u_complete_list)

    def eval_at_sensor_positions(self, sol, pos):
        """
        Evaluates dedalus field at sensor positions.

        Args:
            sol : dist.Field
            Field.
            pos : list
                Sensor positions.

        Returns:
        temp : np.ndarray
            Field 'sol' evaluated at sensor positions 'pos'.

        """
        temp = np.zeros_like(pos)

        for i, sensor_pos in enumerate(pos):
            sol_int = sol(x=sensor_pos)
            temp[i] = sol_int.evaluate()['g'].item()

        return temp
