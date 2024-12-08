import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from scipy.integrate import odeint
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation



class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
    return arrow

setattr(Axes3D, 'arrow3D', _arrow3D)

def slope_field(t, x, diffeq, 
                units = 'xy', 
                angles = 'xy',
                scale_units = 'x',
                scale = None, 
                color = 'black',
                ax = None,
                **args):
    """Plots slope field given an ode
    
    Given an ode of the form: dx/dt = f(t, x), plot a slope field (aka direction field) for given t and x arrays. 
    Extra arguments are passed to matplotlib.pyplot.quiver

    Parameters
    ----------
    
    t : array
        The independent variable range
        
    x : array
        The dependent variable range
    
    diffeq : function
        The function f(t,x) = dx/dt

    args:
        Additional arguments are aesthetic choices passed to pyplot.quiver function
    
    ax : pyplot plotting axes
        Optional existing axis to pass to function

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """
    if (ax is None):
        fig, ax = plt.subplots()
    if scale is not None:
        scale = 1/scale
    T, X = np.meshgrid(t, x)  # create rectangular grid with points
    slopes = diffeq(T, X)
    dt = np.ones(slopes.shape)  # dt = an array of 1's with same dimension as diffeq
    dxu = slopes / np.sqrt(dt**2 + slopes**2)  # normalize dx
    dtu = dt / np.sqrt(dt**2 + slopes**2)  # normalize dt
    ax.quiver(T, X, dtu, dxu,  # Plot a 2D field of arrows
               units = units,  
               angles = angles,  # each arrow has direction from (t,x) to (t+dt, x+dx)
               scale_units = 'x',
               scale = scale,  # sets the length of each arrow from user inputs
               color = color,
               **args)  # sets the color of each arrow from user inputs
    
    return ax


def plot_sol(t, x, diffeq, t0, x0, ax = None, npts=100, clear=False):
    """Plot Slope field and estimated solution given initial conidtion
    
    Given an ode of the form: dx/dt = f(t, x), plot a slope field (aka direction field) for given t and x arrays. 
    Given an initial condition x(t0) = 0, plot a solution line estimated with `odeint`.
    Extra arguments are passed to matplotlib.pyplot.quiver

    Parameters
    ----------
    
    t : array
        The independent variable range
        
    x : array
        The dependent variable range
    
    diffeq : function
        The function f(t,x) = dx/dt

    t0 : float
        Initial time

    x0 : float
        Initial condition at t0
        
    ax : pyplot plotting axes
    
    args:
        Additional arguments are aesthetic choices passed to pyplot.quiver function
    

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """
    
    if (ax is None):
        fig, ax = plt.subplots()
    if clear:
        ax.cla()

    slope_field(t, x, diffeq, color='grey', ax = ax)
    
    # Plot solution
    tt = np.linspace(t0, t.max(), 100)
    sol = odeint(diffeq, x0, tt, tfirst=True)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    ax.plot(tt, sol)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # plt.show()
    return ax
    
    
# Euler Method
def euler_method(diffeq, t0, x0, dt, n):
    # Assuming diffeq is passed as a function of (t, x)
    v = np.zeros(n+1)  # set each x_i by 0 at first
    v[0] = x0  # set first value to x_0
    
    for i in range(0, n):
        v[i+1] = v[i] + dt * diffeq(t0 + i*dt, v[i])  # Euler's method formula
    return v


# Plot Slope field and solution given analytical solution
def plot_euler(t, x, diffeq, t0, x0, dt, n=None, ax=None, clear=False):
    
    if (ax is None):
        fig, ax = plt.subplots()
    if clear:
        ax.cla()

    slope_field(t, x, diffeq, color='grey', ax = ax)

    # Plot exact
    tt = np.linspace(t0,t.max(),100)
    sol = odeint(diffeq, x0, tt, tfirst=True)
    ax.plot(tt, sol)
    
    # Store limits to keep approx from changing
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Plot approx
    if n is None:
        n = int(t[-1]/dt)
    tt2 = np.linspace(t0, n*dt, n+1)
    ax.plot(tt2, euler_method(diffeq, t[0], x0, dt, n), ':', 
             marker='s')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # plt.show()
    return ax


# Plot Slope field and solution given analytical solution
def plot_dt(t, x, diffeq, t0, x0, dt, nsteps, color = 'blue', ax=None, npts=100, clear=False):
    
    if (ax is None):
        fig, ax = plt.subplots()
    if clear:
        ax.cla()
    slope_field(t, x, diffeq, color='grey', ax = ax)
    
    # Store limits to keep approx from changing
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Plot vector
    # scale=1/dt makes the vector 
    tt = np.linspace(t0, t0 + dt*(nsteps-1), nsteps)
    xx = euler_method(diffeq, t[0], x0, dt, nsteps)
    
    for i in range(0, nsteps):
        slope_field(tt[i], xx[i], diffeq, color = color, scale=dt, ax = ax)  
    
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # plt.show()
    return ax


def phase_portrait(v1, v2, diffeq,
                  *params,
                  color='black',
                  ax=None):
    """Plots phase portrait given a dynamical system
    
    Given a dynamical system of the form dXdt=f(X,t,...), plot the phase portrait of the system.

    Parameters
    ----------
    
    v1 : array
        The range of the first variable
        
    v2 : array
        The range of the second variable
    
    diffeq : function
        The function dXdt = f(X,t,...)

    params:
        System parameters to be passed to diffeq
    
    ax : pyplot plotting axes
        Optional existing axis to pass to function

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(11.7,8.3))

    V1, V2 = np.meshgrid(v1, v2)  # create rectangular grid with points
    t = 0 # start time
    u, w = np.zeros(V1.shape), np.zeros(V2.shape) # initiate values for quiver arrows
    NI, NJ = V1.shape
    
    for i in range(NI):
        for j in range(NJ):
            xcoord = V1[i, j]
            ycoord = V2[i, j]
            vprime = diffeq(t, [xcoord, ycoord], *params)
            u[i,j] = vprime[0]
            w[i,j] = vprime[1]
    
    r = np.power(np.add(np.power(u,2), np.power(w,2)),0.5)
    r = np.where(r==0, 1, r) 
    
    Q = ax.quiver(V1, V2, u/r, w/r, scale=110, scale_units='width', headwidth=3, headlength=2, headaxislength=1, color=color)
    
    return ax


def phase_portrait_3d(v1, v2, v3, diffeq,
                  *params,
                  color='black',
                  ax=None):
    """Plots phase portrait given a dynamical system
    
    Given a dynamical system of the form dXdt=f(X,t,...), plot the phase portrait of the system.

    Parameters
    ----------
    
    v1 : array
        The range of the first variable
        
    v2 : array
        The range of the second variable
    
    diffeq : function
        The function dXdt = f(X,t,...)

    params:
        System parameters to be passed to diffeq

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """

    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')

    V1, V2, V3 = np.meshgrid(v1, v2, v3)  # create rectangular grid with points
    t = 0 # start time
    u, v, w = np.zeros(V1.shape), np.zeros(V2.shape), np.zeros(V3.shape) # initiate values for quiver arrows
    NI, NJ, NK = V1.shape
    
    for i in range(NI):
        for j in range(NJ):
            for k in range(NK):
                xcoord = V1[i, j, k]
                ycoord = V2[i, j, k]
                zcoord = V3[i, j, k]
                vprime = diffeq(t, [xcoord, ycoord, zcoord], *params)
                u[i,j,k] = vprime[0]
                v[i,j,k] = vprime[1]
                w[i,j,k] = vprime[2]
    
    r = np.power(np.add(np.power(u,2), np.power(v,2), np.power(w,2)),0.5)
    r = np.where(r==0, 1, r) 
    
    Q = ax.quiver(V1, V2, V3, u/r, v/r, w/r, color=color)
    
    return ax

def plot_phase_sol_manta(v1, v2, vs_all, components, diffeq, tfirst, xlim, ylim,
                  *params,
                  color='black',
                  markersize=5,
                  linewidth = 2,
                  add_phase_plane = True):
    
    """Plots phase portrait with solution given a dynamical system and initial conditions
    
    Given a dynamical system of the form dXdt=f(X,t,...), and additionally two initial conditions, plot the phase portrait of the system and solution.

    Parameters
    ----------
    
    v1 : array
        The range of the first variable
        
    v2 : array
        The range of the second variable

    vs : array
        The integrated solution

    components : list
        Gives the components of the solution vector to plot. E.g. for 3d x, y, z: components = (0,1) would plot x and y.
    
    diffeq : function
        The function dXdt = f(X,t,...)

    params: Additional arguments
        System parameters to be passed to diffeq

    tfirst: float
        The first time to begin plotting
        
    v1_0: float
        Initial condition for v1 variable
        
    v2_0: float
        Initial condition for v2 variable
    
    add_phase_plane: bool
        Add slope field plot. Only makes sense if 2-D system

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """
   
    fig, ax = plt.subplots(figsize=(11.7,8.3))
   
    if add_phase_plane:
        phase_portrait(v1, v2, diffeq, ax=ax, *params)

    for idx, vs in enumerate(vs_all):

        # xs = vs.y[components[0]]
        # ys = vs.y[components[1]]
        # ts = vs.t
        xs = vs[0]
        ys = vs[1]
        ts = vs[2]

        if tfirst is not None:
            idx = np.where(ts > tfirst)
            xs = xs[idx]
            ys = ys[idx]

        if idx == 0:
            linestyle = 'solid'
            color = 'tab:orange'
        else:
            linestyle = 'dashed'
            color = 'blue'
        ax.plot(xs, ys, linewidth=linewidth, linestyle=linestyle, alpha=0.8, color=color, label='norm ODE soln_'+str(idx+1))  
        ax.legend()
        ax.plot([xs[0]], [ys[0]], 'go', markersize=markersize, alpha=0.5)  
        ax.plot([xs[-1]], [ys[-1]], 'ro', markersize=markersize, alpha=0.5)  
        ax.plot([-1], [-1], 'bx', markersize=markersize, alpha=0.5)  
        ax.plot([-1], [1], 'bx', markersize=markersize, alpha=0.5)  
        ax.plot([0], [0], 'bx', markersize=markersize, alpha=0.5)  

    # Label each axis
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.axvline(x=-1.0, color = 'k', linestyle = ':')
    ax.axhline(y=0.0, color = 'k', linestyle = ':')

    plt.show()
    
    return ax

def plot_phase_sol_lotka(v1, v2, vs_all, components, diffeq, tfirst, xlim, ylim,
                  *params,
                  color='black',
                  markersize=5,
                  linewidth = 2,
                  add_phase_plane = True):
    
    """Plots phase portrait with solution given a dynamical system and initial conditions
    
    Given a dynamical system of the form dXdt=f(X,t,...), and additionally two initial conditions, plot the phase portrait of the system and solution.

    Parameters
    ----------
    
    v1 : array
        The range of the first variable
        
    v2 : array
        The range of the second variable

    vs : array
        The integrated solution

    components : list
        Gives the components of the solution vector to plot. E.g. for 3d x, y, z: components = (0,1) would plot x and y.
    
    diffeq : function
        The function dXdt = f(X,t,...)

    params: Additional arguments
        System parameters to be passed to diffeq

    tfirst: float
        The first time to begin plotting
        
    v1_0: float
        Initial condition for v1 variable
        
    v2_0: float
        Initial condition for v2 variable
    
    add_phase_plane: bool
        Add slope field plot. Only makes sense if 2-D system

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """
   
    fig, ax = plt.subplots(figsize=(11.7,8.3))
   
    if add_phase_plane:
        phase_portrait(v1, v2, diffeq, ax=ax, *params)

    for idx, vs in enumerate(vs_all):

        # xs = vs.y[components[0]]
        # ys = vs.y[components[1]]
        # ts = vs.t
        xs = vs[0]
        ys = vs[1]
        ts = vs[2]

        if tfirst is not None:
            idx = np.where(ts > tfirst)
            xs = xs[idx]
            ys = ys[idx]

        if idx == 0:
            linestyle = 'solid'
            color = 'tab:orange'
        else:
            linestyle = 'dashed'
            color = 'blue'
        ax.plot(xs, ys, linewidth=linewidth, linestyle=linestyle, alpha=0.8, color=color, label='norm ODE soln_'+str(idx+1))  
        ax.legend()
        ax.plot([xs[0]], [ys[0]], 'go', markersize=markersize, alpha=0.5)  
        ax.plot([xs[-1]], [ys[-1]], 'ro', markersize=markersize, alpha=0.5)  
    ax.plot([0], [0], 'b*', markersize=markersize, alpha=0.5)  
    ax.plot([3], [0], 'b*', markersize=markersize, alpha=0.5)  
    ax.plot([0], [2], 'b*', markersize=markersize, alpha=0.5)  
    ax.plot([1], [1], 'b*', markersize=markersize, alpha=0.5)  

    # Label each axis
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.show()
    
    return ax

def plot_phase_sol(v1, v2, vs, components, diffeq, tfirst, xlim, ylim,
                  *params,
                  color='black',
                  markersize=5,
                  linewidth = 2,
                  add_phase_plane = True):
    
    """Plots phase portrait with solution given a dynamical system and initial conditions
    
    Given a dynamical system of the form dXdt=f(X,t,...), and additionally two initial conditions, plot the phase portrait of the system and solution.

    Parameters
    ----------
    
    v1 : array
        The range of the first variable
        
    v2 : array
        The range of the second variable

    vs : array
        The integrated solution

    components : list
        Gives the components of the solution vector to plot. E.g. for 3d x, y, z: components = (0,1) would plot x and y.
    
    diffeq : function
        The function dXdt = f(X,t,...)

    params: Additional arguments
        System parameters to be passed to diffeq

    tfirst: float
        The first time to begin plotting
        
    v1_0: float
        Initial condition for v1 variable
        
    v2_0: float
        Initial condition for v2 variable
    
    add_phase_plane: bool
        Add slope field plot. Only makes sense if 2-D system

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """
   
    fig, ax = plt.subplots(figsize=(11.7,8.3))
   
    if add_phase_plane:
        phase_portrait(v1, v2, components, diffeq, ax=ax, *params)

    xs = vs.y[components[0]]
    ys = vs.y[components[1]]
    ts = vs.t

    if tfirst is not None:
        idx = np.where(ts > tfirst)
        xs = xs[idx]
        ys = ys[idx]

    ax.plot(xs, ys, linewidth=linewidth, label='norm ODE soln')  
    ax.legend()
    ax.plot([xs[0]], [ys[0]], 'go', markersize=markersize, alpha=0.5)  
    ax.plot([xs[-1]], [ys[-1]], 'ro', markersize=markersize, alpha=0.5)  

    # Label each axis
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.show()
    
    return ax


def plot_phase_sol_3d_chen(v1, v2, v3, vs, components, diffeq, tfirst, xlim, ylim, zlim, xlabel='x', ylabel='y', zlabel='z',
                  *params,
                  v_extra=None,
                  color='black',
                  ax=None,
                  markersize=7,
                  shading=0.8,
                  linewidth = 2,
                  plot_normalized=False,
                  add_phase_plane = True):
    
    """Plots phase portrait with solution given a dynamical system and initial conditions
    
    Given a dynamical system of the form dXdt=f(X,t,...), and additionally two initial conditions, plot the phase portrait of the system and solution.

    Parameters
    ----------
    
    v1 : array
        The range of the first variable
        
    v2 : array
        The range of the second variable

    vs : array
        The integrated solution

    components : list
        Gives the components of the solution vector to plot. E.g. for 3d x, y, z: components = (0,1) would plot x and y.
    
    diffeq : function
        The function dXdt = f(X,t,...)

    params: Additional arguments
        System parameters to be passed to diffeq

    tfirst: float
        The first time to begin plotting
        
    v1_0: float
        Initial condition for v1 variable
        
    v2_0: float
        Initial condition for v2 variable
    
    add_phase_plane: bool
        Add slope field plot. Only makes sense if 3-D system

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plt.gcf().set_size_inches(11.7,8.3) 
    
    if add_phase_plane:
        phase_portrait_3d(v1, v2, v3, diffeq, ax=ax, *params)

    xs = vs.y[components[0]]
    ys = vs.y[components[1]]
    zs = vs.y[components[2]]
    ts = vs.t

    len_v = np.sqrt(xs*xs + ys*ys + zs*zs)

    nx = xs/len_v
    ny = ys/len_v
    nz = zs/len_v

    if plot_normalized:
        xs = nx
        ys = ny
        zs = nz

    if tfirst is not None:
        idx = np.where(ts > tfirst)
        xs = xs[idx]
        ys = ys[idx]
        zs = zs[idx]

    # Dynamical system info
    a = 35.0
    b = 3.0
    c = 28.0
    e1_x = np.sqrt(b*(2*c - a))
    e1_y = np.sqrt(b*(2*c - a))
    e1_z = 2*c - a
    e2_x = - e1_x
    e2_y = - e1_y
    e2_z = e1_z
    
    ax.plot(xs, ys, zs, linewidth=linewidth, label='norm ODE soln')
    ax.plot(e1_x, e1_z, e1_y, 'k*', markersize=markersize/2, alpha=shading, label='eq1')  
    ax.plot(e2_x, e2_z, e2_y, 'kx', markersize=markersize/2, alpha=shading, label='eq2')  
    ax.plot([xs[0]], [ys[0]], [zs[0]], 'go', markersize=markersize/2, alpha=shading, label='start')  
    ax.plot([xs[-1]], [ys[-1]], [zs[-1]], 'ro', markersize=markersize/2, alpha=shading, label='final')  
   
    # ax.arrow3D(0,0,0,
    #         xs[-1],ys[-1],zs[-1],
    #         mutation_scale=20,
    #         arrowstyle="-|>",
    #         linestyle='solid', label='R vector')
    
    if v_extra is not None:
        wx = v_extra[0]
        wy = v_extra[1]
        wz = v_extra[2]
        ax.arrow3D(0,0,0, wx[-1], wy[-1], wz[-1], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='m', label='w')
    
    ax.legend()
    
    # Label each axis
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # Set each axis limits
    if plot_normalized:
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5)) 
        ax.set_zlim((-1.5, 1.5)) 
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim) 
        ax.set_zlim(zlim) 



    # fig.update_layout(
    #     scene = dict(
    #         xaxis = dict(nticks=4, range=[-100,100],),
    #                     yaxis = dict(nticks=4, range=[None, 100],),
    #                     zaxis = dict(nticks=4, range=[-100, None],),),
    #     width=700,
    #     margin=dict(r=20, l=10, b=10, t=10))
        
    return ax

def plot_phase_sol_3d_quad_lorenz(v1, v2, v3, vs_all, components, diffeq, tfirst, xlim, ylim, zlim, params=None, xlabel='x', ylabel='y', zlabel='z',
                  v_extra=None,
                  color='black',
                  ax=None,
                  markersize=7,
                  shading=0.8,
                  linewidth = 2,
                  plot_normalized=False,
                  add_phase_plane = True):
    
    """Plots phase portrait with solution given a dynamical system and initial conditions
    
    Given a dynamical system of the form dXdt=f(X,t,...), and additionally two initial conditions, plot the phase portrait of the system and solution.

    Parameters
    ----------
    
    v1 : array
        The range of the first variable
        
    v2 : array
        The range of the second variable

    vs : array
        The integrated solution

    components : list
        Gives the components of the solution vector to plot. E.g. for 3d x, y, z: components = (0,1) would plot x and y.
    
    diffeq : function
        The function dXdt = f(X,t,...)

    params: Additional arguments
        System parameters to be passed to diffeq

    tfirst: float
        The first time to begin plotting
        
    v1_0: float
        Initial condition for v1 variable
        
    v2_0: float
        Initial condition for v2 variable
    
    add_phase_plane: bool
        Add slope field plot. Only makes sense if 3-D system

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plt.gcf().set_size_inches(11.7,8.3) 
    
    if add_phase_plane:
        phase_portrait_3d(v1, v2, v3, diffeq, ax=ax, *params)

    if params is not None:
        # Dynamical system info
        b = params[1]
        c = params[2]
        e2_x = np.power(b*c, 1./3.)
        e2_y = np.power(b*c, 1./3.)
        e2_z = c

    for idx, vs in enumerate(vs_all):

        # xs = vs.y[components[0]]
        # ys = vs.y[components[1]]
        # ts = vs.t
        xs = vs[0]
        ys = vs[1]
        zs = vs[2]
        ts = vs[3]

        if tfirst is not None:
            idx = np.where(ts > tfirst)
            xs = xs[idx]
            ys = ys[idx]
            zs = zs[idx]
        
        if idx == 0:
            linestyle = 'solid'
            color = 'tab:orange'
            ax.plot(0.0, 0.0, 0.0, 'k*', markersize=markersize/2, alpha=shading, label='eq1')  
            ax.plot(e2_x, e2_y, e2_z, 'kx', markersize=markersize/2, alpha=shading, label='eq2')  
            ax.plot([xs[0]], [ys[0]], [zs[0]], 'go', markersize=markersize/2, alpha=shading, label='start')  
            ax.plot([xs[-1]], [ys[-1]], [zs[-1]], 'ro', markersize=markersize/2, alpha=shading, label='final')  
        else:
            linestyle = 'dashed'
            color = 'blue'
        ax.plot(xs, ys, zs, linewidth=linewidth, linestyle=linestyle, alpha=0.8, color=color, label='norm ODE soln_'+str(idx+1))  
        ax.legend()

    # Label each axis
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    plt.show()
    
    return ax


def plot_phase_sol_3d_waves(v1, v2, v3, vs, components, diffeq, tfirst, xlim, ylim, zlim, xlabel='x', ylabel='y', zlabel='z',
                  *params,
                  v_extra=None,
                  color='black',
                  ax=None,
                  markersize=7,
                  shading=0.8,
                  linewidth = 2,
                  plot_normalized=False,
                  add_phase_plane = True):
    
    """Plots phase portrait with solution given a dynamical system and initial conditions
    
    Given a dynamical system of the form dXdt=f(X,t,...), and additionally two initial conditions, plot the phase portrait of the system and solution.

    Parameters
    ----------
    
    v1 : array
        The range of the first variable
        
    v2 : array
        The range of the second variable

    vs : array
        The integrated solution

    components : list
        Gives the components of the solution vector to plot. E.g. for 3d x, y, z: components = (0,1) would plot x and y.
    
    diffeq : function
        The function dXdt = f(X,t,...)

    params: Additional arguments
        System parameters to be passed to diffeq

    tfirst: float
        The first time to begin plotting
        
    v1_0: float
        Initial condition for v1 variable
        
    v2_0: float
        Initial condition for v2 variable
    
    add_phase_plane: bool
        Add slope field plot. Only makes sense if 3-D system

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plt.gcf().set_size_inches(11.7,8.3) 
    
    if add_phase_plane:
        phase_portrait_3d(v1, v2, v3, diffeq, ax=ax, *params)

    xs = vs.y[components[0]]
    ys = vs.y[components[1]]
    zs = vs.y[components[2]]
    ts = vs.t

    len_v = np.sqrt(xs*xs + ys*ys + zs*zs)

    nx = xs/len_v
    ny = ys/len_v
    nz = zs/len_v

    if plot_normalized:
        xs = nx
        ys = ny
        zs = nz

    if tfirst is not None:
        idx = np.where(ts > tfirst)
        xs = xs[idx]
        ys = ys[idx]
        zs = zs[idx]

    # Dynamical system info
    a = 35.0
    b = 3.0
    c = 28.0
    e1_x = np.sqrt(b*(2*c - a))
    e1_y = np.sqrt(b*(2*c - a))
    e1_z = 2*c - a
    e2_x = - e1_x
    e2_y = - e1_y
    e2_z = e1_z
    
    ax.plot(xs, ys, zs, linewidth=linewidth, label='norm ODE soln')
    ax.plot(e1_x, e1_z, e1_y, 'k*', markersize=markersize/2, alpha=shading, label='eq1')  
    ax.plot(e2_x, e2_z, e2_y, 'kx', markersize=markersize/2, alpha=shading, label='eq2')  
    ax.plot([xs[0]], [ys[0]], [zs[0]], 'go', markersize=markersize/2, alpha=shading, label='start')  
    ax.plot([xs[-1]], [ys[-1]], [zs[-1]], 'ro', markersize=markersize/2, alpha=shading, label='final')  
   
    # ax.arrow3D(0,0,0,
    #         xs[-1],ys[-1],zs[-1],
    #         mutation_scale=20,
    #         arrowstyle="-|>",
    #         linestyle='solid', label='R vector')
    
    if v_extra is not None:
        wx = v_extra[0]
        wy = v_extra[1]
        wz = v_extra[2]
        ax.arrow3D(0,0,0, wx[-1], wy[-1], wz[-1], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='m', label='w')
    
    ax.legend()
    
    # Label each axis
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # Set each axis limits
    if plot_normalized:
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5)) 
        ax.set_zlim((-1.5, 1.5)) 
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim) 
        ax.set_zlim(zlim) 



    # fig.update_layout(
    #     scene = dict(
    #         xaxis = dict(nticks=4, range=[-100,100],),
    #                     yaxis = dict(nticks=4, range=[None, 100],),
    #                     zaxis = dict(nticks=4, range=[-100, None],),),
    #     width=700,
    #     margin=dict(r=20, l=10, b=10, t=10))
        
    return ax


def plot_phase_sol_3d(v1, v2, v3, vs, components, diffeq, tfirst, xlim, ylim, zlim, xlabel='x', ylabel='y', zlabel='z',
                  *params,
                  v_extra=None,
                  color='black',
                  ax=None,
                  markersize=5,
                  shading=0.8,
                  linewidth = 2,
                  plot_normalized=False,
                  add_phase_plane = True,
                  title=None):
    
    """Plots phase portrait with solution given a dynamical system and initial conditions
    
    Given a dynamical system of the form dXdt=f(X,t,...), and additionally two initial conditions, plot the phase portrait of the system and solution.

    Parameters
    ----------
    
    v1 : array
        The range of the first variable
        
    v2 : array
        The range of the second variable

    vs : array
        The integrated solution

    components : list
        Gives the components of the solution vector to plot. E.g. for 3d x, y, z: components = (0,1) would plot x and y.
    
    diffeq : function
        The function dXdt = f(X,t,...)

    params: Additional arguments
        System parameters to be passed to diffeq

    tfirst: float
        The first time to begin plotting
        
    v1_0: float
        Initial condition for v1 variable
        
    v2_0: float
        Initial condition for v2 variable
    
    add_phase_plane: bool
        Add slope field plot. Only makes sense if 3-D system

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plt.gcf().set_size_inches(11.7,8.3) 
    
    if add_phase_plane:
        phase_portrait_3d(v1, v2, v3, diffeq, ax=ax, *params)

    xs = vs.y[components[0]]
    ys = vs.y[components[1]]
    zs = vs.y[components[2]]
    ts = vs.t

    len_v = np.sqrt(xs*xs + ys*ys + zs*zs)

    nx = xs/len_v
    ny = ys/len_v
    nz = zs/len_v

    if plot_normalized:
        xs = nx
        ys = ny
        zs = nz

    if tfirst is not None:
        idx = np.where(ts > tfirst)
        xs = xs[idx]
        ys = ys[idx]
        zs = zs[idx]
    
    # ax.plot(xs, ys, zs, linewidth=linewidth)
    if plot_normalized:
        ax.plot(xs, ys, zs, linewidth=linewidth, label='norm ODE soln')
    else:
        ax.plot(xs, ys, zs, linewidth=linewidth, label='ODE soln')
    # ax.plot(0, 0, 0, 'k*', markersize=markersize/2, alpha=shading, label='origin')  
    # ax.plot([xs[0]], [ys[0]], [zs[0]], 'go', markersize=markersize, alpha=shading, label='start')  
    # ax.plot([xs[-1]], [ys[-1]], [zs[-1]], 'ro', markersize=markersize, alpha=shading, label='final')  
    ax.plot(0, 0, 0, 'k*', markersize=markersize/2, alpha=shading)  
    ax.plot([xs[0]], [ys[0]], [zs[0]], 'go', markersize=markersize, alpha=shading)  
    ax.plot([xs[-1]], [ys[-1]], [zs[-1]], 'ro', markersize=markersize, alpha=shading)  
   
    if plot_normalized:
        ax.arrow3D(0,0,0,
                xs[-1],ys[-1],zs[-1],
                mutation_scale=20,
                arrowstyle="-|>",
                linestyle='solid', color='k', label=r'$\hat E$')
                # linestyle='solid', color='k', label='E')
    else:
        ax.arrow3D(0,0,0,
                xs[-1],ys[-1],zs[-1],
                mutation_scale=20,
                arrowstyle="-|>",
                linestyle='solid', color='k', label=r'$E$')
                # linestyle='solid', color='k', label='E')
    
    if v_extra is not None:
        w = v_extra[0]
        wx = w[0]
        wy = w[1]
        wz = w[2]

        len_w = np.sqrt(wx*wx + wy*wy + wz*wz)
        nwx = wx/len_w
        nwy = wy/len_w
        nwz = wz/len_w
        if plot_normalized:
            wx = nwx
            wy = nwy
            wz = nwz
        ax.arrow3D(0,0,0, wx[-1], wy[-1], wz[-1], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='m', label=r'$\omega$')

        if len(v_extra) == 2:
            B = v_extra[1]
            Bx = B[0]
            By = B[1]
            Bz = B[2]

            len_B = np.sqrt(Bx*Bx + By*By + Bz*Bz)
            nBx = Bx/len_B
            nBy = By/len_B
            nBz = Bz/len_B
            if plot_normalized:
                Bx = nBx
                By = nBy
                Bz = nBz

            ax.arrow3D(0,0,0, Bx[-1], By[-1], Bz[-1], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='y', label='B')
    
    ax.legend(loc=(0.705, 0.64))
    
    # Label each axis
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)

    # Set each axis limits
    if plot_normalized:
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5)) 
        ax.set_zlim((-1.5, 1.5)) 
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim) 
        ax.set_zlim(zlim) 
    
    return ax


def plot_animation_soln_3d(v1, v2, v3, vs, components, diffeq, v_extra, plot_normalized, tfirst, xlim, ylim, zlim, xlabel='x', ylabel='y', zlabel='z',
                  *params,
                  interval=200,
                  color='black',
                  ax=None,
                  markersize=7,
                  shading=0.8,
                  linewidth=2,
                  add_phase_plane=False,
                  title=None,
                  path=None):
    
    """Plots phase portrait with solution given a dynamical system and initial conditions
    
    Given a dynamical system of the form dXdt=f(X,t,...), and additionally two initial conditions, plot the phase portrait of the system and solution.

    Parameters
    ----------
    
    v1 : array
        The range of the first variable
        
    v2 : array
        The range of the second variable

    vs : array
        The integrated solution

    components : list
        Gives the components of the solution vector to plot. E.g. for 3d x, y, z: components = (0,1) would plot x and y.
    
    diffeq : function
        The function dXdt = f(X,t,...)

    v_extra : array
        An extra array or vector to add to the plot

    params: Additional arguments
        System parameters to be passed to diffeq

    tfirst: float
        The first time to begin plotting
        
    v1_0: float
        Initial condition for v1 variable
        
    v2_0: float
        Initial condition for v2 variable
    
    add_phase_plane: bool
        Add slope field plot. Only makes sense if 3-D system

    Returns
    -------
    out : ax
        plotting axis with formatted quiver plot
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plt.gcf().set_size_inches(11.7,8.3) 
    
    if add_phase_plane:
        phase_portrait_3d(v1, v2, v3, diffeq, ax=ax, *params)

    xs = vs.y[components[0]]
    ys = vs.y[components[1]]
    zs = vs.y[components[2]]
    ts = vs.t

    len_v = np.sqrt(xs*xs + ys*ys + zs*zs)

    nx = xs/len_v
    ny = ys/len_v
    nz = zs/len_v

    if plot_normalized:
        xs = nx
        ys = ny
        zs = nz

    if tfirst is not None:
        idx = np.where(ts > tfirst)
        xs = xs[idx]
        ys = ys[idx]
        zs = zs[idx]


    def update_vector(i, xs, ys, zs, wx, wy, wz, ax):
        global pts
        global arrows
        global arrows2
        global arrows3
        global second_i
     
        if plot_normalized:
            ax.plot(xs[:i], ys[:i], zs[:i], linewidth=linewidth, color='b', label='norm ODE soln')
        else:
            ax.plot(xs[:i], ys[:i], zs[:i], linewidth=linewidth, color='b', label='ODE soln')

        if i != 0:
            pts.remove()           
            pts, = ax.plot([xs[:i][-1]], [ys[:i][-1]], [zs[:i][-1]], 'ro', markersize=markersize, alpha=shading)
            arrows.remove()
            if plot_normalized:
                arrows = ax.arrow3D(0,0,0, xs[:i][-1], ys[:i][-1], zs[:i][-1], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='k', label=r'$\hat E$')
            else:
                arrows = ax.arrow3D(0,0,0, xs[:i][-1], ys[:i][-1], zs[:i][-1], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='k', label='E')
            arrows2.remove()
            arrows2 = ax.arrow3D(0,0,0, wx[i], wy[i], wz[i], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='m', label=r'$\omega$')
            arrows3.remove()
            arrows3 = ax.arrow3D(0,0,0, Bx[i], By[i], Bz[i], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='y', label='B')

        if i == 0:
            if second_i == 0:
                ax.legend()
            second_i = 1

        return ax
    
    # Label each axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Set each axis limits
    if plot_normalized:
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5)) 
        ax.set_zlim((-1.5, 1.5)) 
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim) 
        ax.set_zlim(zlim) 

    ax.plot(0, 0, 0, 'k*', markersize=markersize, alpha=shading) 
    ax.plot([xs[0]], [ys[0]], [zs[0]], 'go', markersize=markersize, alpha=shading)   

    global pts
    pts, = plt.plot([], [], [], marker='o', color='r') 

    global arrows
    if plot_normalized:
        arrows = ax.arrow3D(0,0,0, xs[0], ys[0], zs[0], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='k', label=r'$\hat E$')
    else:
        arrows = ax.arrow3D(0,0,0, xs[0], ys[0], zs[0], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='k', label='E')

    if v_extra is not None:
        w = v_extra[0]
        global arrows2
        wx = w[0]
        wy = w[1]
        wz = w[2]
        len_w = np.sqrt(wx*wx + wy*wy + wz*wz)
        nwx = wx/len_w
        nwy = wy/len_w
        nwz = wz/len_w
        if plot_normalized:
            wx = nwx
            wy = nwy
            wz = nwz
        arrows2 = ax.arrow3D(0,0,0, wx[0], wy[0], wz[0], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='m', label=r'$\omega$')
        if len(v_extra) == 2:
            B = v_extra[1]
            global arrows3
            Bx = B[0]
            By = B[1]
            Bz = B[2]
            len_B = np.sqrt(Bx*Bx + By*By + Bz*Bz)
            nBx = Bx/len_B
            nBy = By/len_B
            nBz = Bz/len_B
            if plot_normalized:
                Bx = nBx
                By = nBy
                Bz = nBz
            arrows3 = ax.arrow3D(0,0,0, Bx[0], By[0], Bz[0], mutation_scale=20, arrowstyle="-|>", linestyle='solid', color='y', label='B')

    global second_i
    second_i = 0

    ani = animation.FuncAnimation(fig, update_vector, frames=len(wx), fargs=(xs, ys, zs, wx, wy, wz, ax), interval=interval, cache_frame_data=False, repeat=False)
    plt.show()
    # Uncomment these to save
    # writergif = animation.PillowWriter(fps=100) 
    # ani.save("C:\\Users\\jelto\\Desktop\\" + path, writer=writergif)

    return ax

def plot_single_var_vs_time_manta(t_all, x_all, tlim, xlim, xlabel, ylabel):
    fig, ax=plt.subplots()
    plt.xlim(tlim)
    plt.ylim(xlim)
    for i in range(len(t_all)):
        t = t_all[i]
        x = x_all[i]
        if i == 0:
            linestyle = 'solid'
            color = 'tab:orange'
        else:
            linestyle = 'dashed'
            color = 'blue' 
        plt.plot(t, x, linewidth=2, linestyle=linestyle, alpha=0.8, color=color, label=ylabel[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel[0])
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    plt.legend()
    plt.show()

    return None

def plot_single_var_vs_time_lotka(t_all, x_all, tlim, xlim, xlabel, ylabel):
    fig, ax=plt.subplots()
    plt.xlim(tlim)
    plt.ylim(xlim)
    for i in range(len(t_all)):
        t = t_all[i]
        x = x_all[i]
        if i == 0:
            linestyle = 'solid'
            color = 'tab:orange'
        else:
            linestyle = 'dashed'
            color = 'blue' 
        plt.plot(t, x, linewidth=2, linestyle=linestyle, alpha=0.8, color=color, label=ylabel[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel[0])
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    plt.legend()
    plt.show()

    return None

def plot_single_var_vs_time_chen(t_all, x_all, tlim, xlim, xlabel, ylabel):
    fig, ax=plt.subplots()
    plt.xlim(tlim)
    plt.ylim(xlim)
    for i in range(len(t_all)):
        t = t_all[i]
        x = x_all[i]
        if i == 0:
            linestyle = 'solid'
            color = 'tab:orange'
        else:
            linestyle = 'dashed'
            color = 'blue' 
        plt.plot(t, x, linewidth=2, linestyle=linestyle, alpha=0.8, color=color, label=ylabel[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel[0])
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    plt.legend()
    plt.show()

    return None

def plot_single_var_vs_time_quad_lorenz(t_all, x_all, tlim, xlim, xlabel, ylabel):
    fig, ax=plt.subplots()
    plt.xlim(tlim)
    plt.ylim(xlim)
    for i in range(len(t_all)):
        t = t_all[i]
        x = x_all[i]
        if i == 0:
            linestyle = 'solid'
            color = 'tab:orange'
        else:
            linestyle = 'dashed'
            color = 'blue' 
        plt.plot(t, x, linewidth=2, linestyle=linestyle, alpha=0.8, color=color, label=ylabel[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel[0])
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    plt.legend()
    plt.show()

    return None

def plot_single_var_vs_time_waves(t_all, x_all, tlim, xlim, xlabel, ylabel):
    fig, ax=plt.subplots()
    plt.xlim(tlim)
    plt.ylim(xlim)
    for i in range(len(t_all)):
        t = t_all[i]
        x = x_all[i]
        if i == 0:
            linestyle = 'solid'
            color = 'tab:orange'
        else:
            linestyle = 'dashed'
            color = 'blue' 
        plt.plot(t, x, linewidth=2, linestyle=linestyle, alpha=0.8, color=color, label=ylabel[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel[0])
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label=r'$2 \pi_h$')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label=r'$\pi_v$')
    plt.legend()
    plt.show()

    return None

def plot_single_var_vs_time(t, x, tlim, xlim, xlabel, ylabel, title=None, add_h=False):
    fig, ax=plt.subplots()
    plt.xlim(tlim)
    plt.ylim(xlim)
    plt.plot(t, x, label=ylabel)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if add_h:
        ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label=r'$2 \pi$')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label=r'$\pi_v$')
    # ax.axvline(x=2*np.pi, color = 'y', linestyle = ':', label=r'$2 \pi_v$')
    # ax.axvline(x=4*np.pi, color = 'k', linestyle = ':', label=r'$4 \pi_v$')
    plt.legend()
    plt.show()

    return None
