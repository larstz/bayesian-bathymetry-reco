from time import time
import numpy as np
import dedalus.public as d3
from dedalus.core.domain import Domain
from .read_left_bc import leftbc

def gaussian_bathymetry(x: np.ndarray, params: tuple[float, float]) -> np.ndarray:
    """Gaussian bathymetry function.
    Args:
        x: The x-coordinate.

    Returns:
        The bathymetry value at x.
    """
    return 0.2*np.exp(-1/params[1]*(x-params[0])**2)


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
        self.H = 0.3 + 0.5 * np.exp(-(domain.x - xbound[1] / 2) ** 2 / 2 ** 2)\
                * 0.05 * np.sin(0.2 * np.pi * domain.x)


class WaterChannelInitialConditions(InitialConditions):
    """Initial conditions for the water channel problem."""
    def __init__(self, domain: CustomDomain, tstart: float=30.):
        super().__init__(domain)
        self.bc_field = domain.dist.Field()
        self.H = 0.3
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
                 params: dict, bcfile: str='/home/lars/DATA/Code/bayesian-inverse-problems/experiment_data/Data_Sensors_orig/With_Bathymetry/Heat1.txt'):
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

#! TODO: Implement left boundary condition


class SWESolver():
    """Shallow water equation solver.

    Attributes:
        xbound: The x-boundary.
        dt: The time step.
        nx: The number of grid points.
        tend: The end time.
        g: The gravity constant.
        kappa: The bottom friction coefficient.
        dealias: The dealiasing factor.

    Methods:
        __init__: Initialize the SWESolver.
        solve: Solve the shallow water equations.
    """

    def __init__(self, xbound: tuple[float, float], dt: float, nx: int, tend: float,
                 g: float=9.81, kappa: float=0.2, dealias: float=2/3, problemtype: str='periodic'):

        self.xbound = xbound
        self.dt = dt
        self.nx = nx
        self.tend = tend
        self.params = {'g': g, 'kappa': kappa, 'dealias': dealias}
        self.problemtype = problemtype

        # Initialize domain, initial conditions, and solver
        if self.problemtype == 'periodic':
            self.domain = PeriodicDomain(self.xbound, self.nx, self.params['dealias'])
            self.initial_conditions = PeriodicInitialConditions(self.domain)
            self.solver = PeriodicSolver(self.domain, self.initial_conditions, self.params)
        elif self.problemtype == 'waterchannel':
            self.domain = WaterChannelDomain(self.xbound, self.nx, self.params['dealias'])
            self.initial_conditions = WaterChannelInitialConditions(self.domain)
            self.solver = WaterChannelSolver(self.domain, self.initial_conditions, self.params)
        else:
            raise ValueError("Invalid problem type")


    def solve(self, b_array: np.ndarray):
        """Solve the shallow water equations.
        Args:
            peak: The peak of the Gaussian bathymetry.

        Returns:
            h_list: The list of water height.
            u_list: The list of velocity.
            t_list: The list of time.
        """

        # Set the initial conditions
        self.initial_conditions.b['g'] = np.squeeze(b_array)
        self.initial_conditions.h['g'] = self.initial_conditions.H\
                                         - self.initial_conditions.b['g']
        self.initial_conditions.u['g'] = 0

        # Set parameters for the solver
        solver = self.solver.get_problem()
        solver.stop_wall_time = 15000
        solver.stop_iteration = int(self.tend/abs(self.dt))+1
        solver.stop_sim_time = self.tend - 1e-13

        self.initial_conditions.change_space_scales(1)

        h_list = [np.copy(self.initial_conditions.h['g'])]
        u_list = [np.copy(self.initial_conditions.u['g'])]
        t_list = [solver.sim_time]

        while solver.proceed:
            solver.step(self.dt)
            if solver.iteration % 1 == 0:
                self.initial_conditions.h.change_scales(1)
                h_list.append(np.copy(self.initial_conditions.h['g']))
                self.initial_conditions.u.change_scales(1)
                u_list.append(np.copy(self.initial_conditions.u['g']))
                t_list.append(solver.sim_time)
                if np.max(self.initial_conditions.h['g']) > 100:
                    break

        self.initial_conditions.change_space_scales(1)
        return np.array(h_list), np.array(u_list), np.array(t_list)

def main():
    """Main function."""
    xbounds = (0., 10.)
    nx =  64
    tend = 10
    timestep = 1e-3
    g = 9.81
    kappa = 0.2
    dealias = 3/2

    solver = SWESolver(xbounds, timestep, nx, tend, g, kappa, dealias, problemtype='waterchannel')
    b = gaussian_bathymetry(solver.domain.x, (5., 1.))

    # time the solver
    start = time()
    h_list, u_list, t_list = solver.solve(b)
    end = time()
    print(f"Time taken: {end-start}")
    print(f"Number of time steps: {len(t_list)}")


if __name__ == "__main__":
    main()