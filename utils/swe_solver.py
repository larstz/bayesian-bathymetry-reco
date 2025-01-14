import numpy as np
import dedalus.public as d3
from dedalus.core.domain import Domain

def gaussian_bathymetry(x: np.ndarray, peak: float) -> np.ndarray:
    """Gaussian bathymetry function.
    Args:
        x: The x-coordinate.

    Returns:
        The bathymetry value at x.
    """
    return 0.2*np.exp(-(x-peak)**2)


class CustomDomain:
    """Domain for the shallow water equations.

    Attributes:
        xcoord: The x-coordinate.
        dist: The distributor.
        xbasis: The x-basis.
        dom: The domain.
        x: The x-grid.
    """
    def __init__(self, xbound: tuple[float, float], nx: int, dealias: float):
        self.xcoord = d3.Coordinate('x')
        self.dist = d3.Distributor(self.xcoord, dtype=np.float64)
        self.xbasis = d3.RealFourier(self.xcoord, size=nx, bounds=xbound, dealias=dealias)
        self.dom = Domain(self.dist, bases=[self.xbasis])
        self.x = self.dist.local_grid(self.xbasis)

class InitialConditions:
    """Initial conditions for the shallow water equations.

    Attributes:
        h: The water height.
        u: The velocity.
        t: The time.
        b: The bathymetry.
        H: The total water height.
    """
    def __init__(self, domain: CustomDomain, xbound: tuple[float, float]):
        self.h = domain.dist.Field(name='h', bases=domain.xbasis)
        self.u = domain.dist.Field(name='u', bases=domain.xbasis)
        self.t = domain.dist.Field()
        self.b = domain.dist.Field(bases=domain.xbasis)
        self.H = 0.3 + 0.5 * np.exp(-(domain.x - xbound[1] / 2) ** 2 / 2 ** 2) * 0.05 * np.sin(0.2 * np.pi * domain.x)

class Solver:
    """Solver for the shallow water equations.

    Attributes:
        dx: The x-derivative.
        name_dict: The dictionary of names.
        solver: The solver.
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

        problem = d3.IVP([initial_conditions.h, initial_conditions.u],
                         time=initial_conditions.t, namespace=self.name_dict)
        problem.add_equation("dt(h) = -dx(h*u)")
        problem.add_equation("dt(u) + g*dx(h) + kappa*u = - g*dx(b) - u*dx(u)")

        # Build solver
        self.solver = problem.build_solver(d3.RK443)


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
        set_domain: Set the domain.
        set_bathymetry: Set the bathymetry.
        set_initial_conditions: Set the initial conditions.
        set_boundary_conditions: Set the boundary conditions.
        set_solver: Set the solver.
        solve: Solve the shallow water equations.
    """

    def __init__(self, xbound: tuple[float, float], dt: float, nx: int, tend: float,
                 g: float=9.81, kappa: float=0.2, dealias: float=2/3):

        self.xbound = xbound
        self.dt = dt
        self.nx = nx
        self.tend = tend
        self.params = {'g': g, 'kappa': kappa, 'dealias': dealias}
        self.domain = None
        self.initial_conditions =None
        self.solver = None

    def set_domain(self):
        """Set the domain."""
        self.domain = CustomDomain(self.xbound, self.nx, self.params['dealias'])

    def set_initial_conditions(self, ):
        """Set the initial conditions."""
        if self.domain is None:
            self.set_domain()

        self.initial_conditions = InitialConditions(self.domain, self.xbound)

    def set_solver(self):
        """Set the solver."""

        if self.domain is None:
            self.set_domain()

        if self.initial_conditions is None:
            self.set_initial_conditions()

        self.solver = Solver(self.domain, self.initial_conditions, self.params)
        self.solver.solver.stop_wall_time = 15000
        self.solver.solver.stop_iteration = int(self.tend/abs(self.dt))+1
        self.solver.solver.stop_sim_time = self.tend - 1e-13

    def solve(self, peak: float):
        """Solve the shallow water equations.
        Args:
            peak: The peak of the Gaussian bathymetry.

        Returns:
            h_list: The list of water height.
            u_list: The list of velocity.
            t_list: The list of time.
        """
        if self.solver is None:
            self.set_solver()

        self.initial_conditions.b.change_scales(1)
        self.initial_conditions.h.change_scales(1)
        self.initial_conditions.u.change_scales(1)

        self.initial_conditions.b['g'] = gaussian_bathymetry(self.domain.x, peak)
        self.initial_conditions.h['g'] = self.initial_conditions.H\
                                         - self.initial_conditions.b['g']
        self.initial_conditions.u['g'] = 0

        h_list = [np.copy(self.initial_conditions.h['g'])]
        u_list = [np.copy(self.initial_conditions.u['g'])]
        t_list = [self.solver.solver.sim_time]


        while self.solver.solver.proceed:
            self.solver.solver.step(self.dt)
            if self.solver.solver.iteration % 1 == 0:
                self.initial_conditions.h.change_scales(1)
                h_list.append(np.copy(self.initial_conditions.h['g']))
                self.initial_conditions.u.change_scales(1)
                u_list.append(np.copy(self.initial_conditions.u['g']))
                t_list.append(self.solver.solver.sim_time)
                if np.max(self.initial_conditions.h['g']) > 100:
                    break
        return h_list, u_list, t_list

def main():

    xmin = 0
    xmax = 10
    # Nx = 64
    Nx = 130
    T = 10
    timestep = 5e-5
    N = int(T/abs(timestep))+1
    g = 9.81
    kappa = 0.2
    dealias = 3/2

    solver = SWESolver((xmin, xmax), timestep, Nx, T, g, kappa, dealias)

    h_list, u_list, t_list = solver.solve(5)
    print(f"Number of time steps: {len(t_list)}")


if __name__ == "__main__":
    main()