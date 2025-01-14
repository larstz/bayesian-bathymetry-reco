import numpy as np
import dedalus.public as d3
from dedalus.core.domain import Domain
from dedalus.extras.plot_tools import quad_mesh, pad_limits

def gaussian_bathymetry(x: np.ndarray, peak: float) -> np.ndarray:
    """Gaussian bathymetry function.
    Args:
        x: The x-coordinate.

    Returns:
        The bathymetry value at x.
    """
    return 0.2*np.exp(-(x-peak)**2)


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
        self.g = g
        self.kappa = kappa
        self.dealias = dealias

    def set_domain(self):
        """Set the domain."""
        self.xcoord = d3.Coordinate('x')
        self.dist = d3.Distributor(self.xcoord, dtype=np.float64)
        self.xbasis = d3.RealFourier(self.xcoord, size=self.nx, bounds=self.xbound,
                                     dealias=self.dealias)
        dom = Domain(self.dist, bases=[self.xbasis])
        self.x = self.dist.local_grid(self.xbasis)

    def set_initial_conditions(self, ):
        """Set the initial conditions."""
        if not hasattr(self, 'dist'):
            self.set_domain()

        self.h = self.dist.Field(name='h', bases=self.xbasis)
        self.u = self.dist.Field(name='u', bases=self.xbasis)
        self.t = self.dist.Field()
        self.b = self.dist.Field(bases=self.xbasis)
        self.H = 0.3 + 0.5*np.exp(-(self.x-self.xbound[1]/2)**2/2**2)\
                 *0.05*np.sin(0.2*np.pi*self.x)

    def set_solver(self):
        """Set the solver."""
        if not hasattr(self, 'h'):
            self.set_initial_conditions()

        def dx(A): return d3.Differentiate(A, self.xcoord)

        name_dict = {'h': self.h, 'u': self.u, 't': self.t, 'b': self.b,
                     'g': self.g, 'kappa': self.kappa, 'dx': dx}

        self.problem = d3.IVP([self.h, self.u], time=self.t, namespace=name_dict)
        self.problem.add_equation("dt(h) = -dx(h*u)")
        self.problem.add_equation("dt(u) + g*dx(h) + kappa*u = - g*dx(b) - u*dx(u)")

        # Build solver
        self.solver = self.problem.build_solver(d3.RK443)
        self.solver.stop_wall_time = 15000
        self.solver.stop_iteration = int(self.tend/abs(self.dt))+1
        self.solver.stop_sim_time = self.tend - 1e-13

    def solve(self, peak: float):
        """Solve the shallow water equations.
        Args:
            peak: The peak of the Gaussian bathymetry.

        Returns:
            h_list: The list of water height.
            u_list: The list of velocity.
            t_list: The list of time.
        """
        if not hasattr(self, 'solver'):
            self.set_initial_conditions()
            self.set_solver()

        self.b.change_scales(1)
        self.h.change_scales(1)
        self.u.change_scales(1)

        self.b['g'] = gaussian_bathymetry(self.x, peak)
        self.h['g'] = self.H - self.b['g']
        self.u['g'] = 0

        h_list = [np.copy(self.h['g'])]
        u_list = [np.copy(self.u['g'])]
        t_list = [self.solver.sim_time]

        # Main loop
        # logger.info('Starting loop')
        while self.solver.proceed:
            self.solver.step(self.dt)
            if self.solver.iteration % 1 == 0:
                self.h.change_scales(1)
                h_list.append(np.copy(self.h['g']))
                self.u.change_scales(1)
                u_list.append(np.copy(self.u['g']))
                t_list.append(self.solver.sim_time)
                if np.max(self.h['g']) > 100:
                    break
        return h_list, u_list, t_list
