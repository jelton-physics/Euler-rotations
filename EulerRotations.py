import numpy as np
import sklearn
from numpy import linalg
from numpy.linalg import norm
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
import itertools
import nolds
from lyapynov import ContinuousDS, DiscreteDS
from lyapynov import mLCE, LCE, CLV, ADJ

from ode_tools_new import plot_phase_sol, plot_phase_sol_3d, phase_portrait, plot_single_var_vs_time, plot_animation_soln_3d
# from utils.ode_tools import plot_phase_sol, plot_phase_sol_3d, phase_portrait, plot_single_var_vs_time, plot_animation_soln_3d

import sys
np.set_printoptions(threshold=sys.maxsize)
import warnings
warnings.filterwarnings("ignore")

def get_angular_velocity(t, Y, w_version, w_params):

    # # Only for doing impulse
    # global iter_num
    # global wx_curr
    # global wy_curr
    # global wz_curr
    # if Y is not None:
    #     Rx, Ry, Rz = Y

    # # Have to unpack params based on specific w used
    if (w_version == "const_rotation") | (w_version == "back_forth_acc"):
        w1 = w_params[0]

    elif (w_version == "torus_rotation") | (w_version == "experient_d_1"):
        w1 = w_params[0]
        w2 = w_params[1]
        r1 = w_params[2]
        r2 = w_params[3]

    elif (w_version == "experiment_a_1") | (w_version == "experiment_a_2") | (w_version == "experiment_b_1") | (w_version == "experiment_b_2") | (w_version == "experiment_b_3"):
        cc = w_params[0]
        t_end = w_params[1]


    # Experiment a
    if w_version == "experiment_a_1":
        wx = 1.0
        wy = cc*t
        wz = 0.0
    elif w_version == "experiment_a_2":
        wx = -1.0
        wy = -cc*(t_end - t)
        wz = 0.0

    if w_version == "experiment_b_1":
        wx = cc*(t**3)*(-np.cos(1.0/t) + 5.0*t*np.sin(1.0/t))
        wy = cc*(t**3)*(np.sin(1.0/t) + 5.0*t*np.cos(1.0/t))
        wz = 0.0
    elif w_version == "experiment_b_2":
        wx = (1.0/(t*t))*(np.exp(-1.0/t))*(np.sin(1.0/t) - np.cos(1.0/t))
        wy = (1.0/(t*t))*(np.exp(-1.0/t))*(np.sin(1.0/t) + np.cos(1.0/t))
        wz = 0.0
    elif w_version == "experiment_b_3":
        wx = (1.0)*(np.sin(np.exp(t)) - np.cos(np.exp(t)))
        wy = (1.0)*(np.sin(np.exp(t)) + np.cos(np.exp(t)))
        wz = 0.0
        # wx = np.cos(np.exp(t))
        # wy = np.sin(np.exp(t))
        # wz = 0.0

    if w_version == "torus_rotation":
        wx = (r1 + r2*np.cos(w1*t))*np.cos(w2*t)
        wy = (r1 + r2*np.cos(w1*t))*np.sin(w2*t)
        wz = r2*np.sin(w1*t)

    if w_version == "back_forth_acc":
        wx = np.abs(np.cos(w1*t))
        wy = np.abs(np.sin(w1*t))
        wz = 0.0

    if w_version == "experiment_d_1":
        L = 2 + np.cos(w2*t)
        wx = L*(np.cos(w1*t))
        wy = L*(np.sin(w1*t))
        wz = 0.0

    # # Constant along e3
    if w_version == "const_e3":
        wx = 0.0
        wy = 0.0
        wz = 1.0

    # # Constant along another axis
    # wx = 1.0/np.sqrt(2)
    # wy = 0.0
    # wz = 1.0/np.sqrt(2)

    # Rotation about e3
    if w_version == "const_rotation":
        wx = np.cos(w1*t)
        wy = np.sin(w1*t)
        wz = 0.0

    # # Impulse
    # if t < np.pi:
    #     wx_curr = 0.0
    #     wy_curr = 0.0
    #     wz_curr = 1.0
    # # elif iter_num % 100 == 0:
    # elif (iter_num % 100 == 0) & (iter_num < 10000000000000000000000000):
    #     len_R = np.sqrt(Rx*Rx + Ry*Ry + Rz*Rz)
    #     # wx_curr = Rx/len_R - 0.00005
    #     # wy_curr = Ry/len_R + 0.0001
    #     # wz_curr = Rz/len_R + 0.0001
    #     wx_curr = Rx/len_R 
    #     wy_curr = Ry/len_R 
    #     wz_curr = Rz/len_R 
    #     z = 1
    # wx = wx_curr
    # wy = wy_curr
    # wz = wz_curr
    # iter_num += 1

    # # Piecewise 
    if w_version == "piecewise_1":
        if t <= np.pi:
            wx = 1
            wy = 0
            wz = 0
        else:
            wx = np.cos(w*t)
            wy = np.sin(w*t)
            wz = 0.0


    # # Piecewise axis flip
    # if t <= np.pi:
    #     wx = 1
    #     wy = 0
    #     wz = 0
    # else:
    #     wx = 0
    #     wy = 0
    #     wz = 1


    # # Rotation about normalized axis k = (kx, ky, kz), |k| = 1
    # kx = 1.0/np.sqrt(3)
    # ky = 1.0/np.sqrt(3)
    # kz = 1.0/np.sqrt(3)

    # w0x = np.cos(t)
    # w0y = np.sin(t)
    # w0z = 0.0

    # if kz == 1:
    #     print("This is just z, no change")
    #     wx = w0x
    #     wy = w0y
    #     wz = w0z
    # else:
    #     c = kz
    #     s = np.sqrt(1 - kz*kz)
    #     n1 = -ky/s
    #     n2 = kx/s
    #     n3 = 0

    #     K00 = c + n1*n1*(1-c)
    #     K01 = n1*n2*(1-c) - n3*s
    #     K02 = n1*n3*(1-c) + n2*s
    #     K10 = n1*n2*(1-c) + n3*s
    #     K11 = c + n2*n2*(1-c)
    #     K12 = n2*n3*(1-c) - n1*s
    #     K20 = n1*n3*(1-c) - n2*s
    #     K21 = n2*n3*(1-c) + n1*s
    #     K22 = c + n3*n3*(1-c)

    #     K = np.array([[K00, K01, K02], [K10, K11, K12], [K20, K21, K22]])
    
    #     w = np.matmul(K, np.array([w0x, w0y, w0z]))
    #     wx = w[0]
    #     wy = w[1]
    #     wz = w[2]

    ### Other stuff

    # wx = t
    # wy = t
    # wz = 0

    # wx = t
    # wy = t*t
    # wz = 0

    return wx, wy, wz

def eval_auxiliary_func(phi):
    if phi == 0.0:
        val = 1.0
    else:
        val = (phi*(1 + np.cos(phi)))/(2*np.sin(phi))
    return val

def phi_function(phi):
    # Length of rotation
    f_phi = phi
    return f_phi

def inverse_phi_fucntion(len_R):
    phi = len_R
    return phi

def inverse_phi_fucntion_2(phi_g, ts):
  
    # keep_array = np.array([], dtype=bool)
    # times = np.array([])
    # for i in range(phi_glob.shape[0]):
    #     if (phi_glob[i,1] in ts) and (phi_glob[i,1] not in times):
    #         keep_array = np.append(keep_array, True)
    #         times = np.append(times, phi_glob[i,1])
    #     else:
    #         keep_array = np.append(keep_array, False)

    # phi_s = phi_glob[:,0][keep_array]

    time_slice = phi_g[:,[1]]
    _, unq_row_indices = np.unique(time_slice,return_index=True,axis=0)
    phi_g = phi_g[unq_row_indices]

    times_mask = np.isin(phi_g[:,1], ts)
    phi_s = phi_g[:,0][times_mask]

    return phi_s

def get_all_angular_velocities(ts, w_version, w_params):
    wx_s = np.array([])
    wy_s = np.array([])
    wz_s = np.array([])
    for i in range(len(ts)):
        t = ts[i]
        wx, wy, wz = get_angular_velocity(t, Y=None, w_version=w_version, w_params=w_params)
        wx_s = np.append(wx_s, wx)
        wy_s = np.append(wy_s, wy)
        wz_s = np.append(wz_s, wz)
    return wx_s, wy_s, wz_s

def plot_angular_velocity(wx_s, wy_s, wz_s, dim):
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        plt.plot(wx_s, wy_s, wz_s, label="phi")
        # Label each axis
        ax.set_xlabel('wx')
        ax.set_ylabel('wy')
        ax.set_zlabel('wz')
        plt.show()
    elif dim == 2:
        fig, ax=plt.subplots()
        # plt.xlim(tlim)
        # plt.ylim(xlim)
        plt.plot(wy_s, wz_s, label="phi")
        plt.xlabel('wy')
        plt.ylabel('wz')
        plt.show()


def power_spectrum_auto(phi, time_step, xlim):

    # The autocorrelation
    def autocorrelation(ts):
        
        yunbiased = ts-np.mean(ts)
        ynorm = np.sum(yunbiased**2)
        acor = np.correlate(yunbiased, yunbiased, "same")/ynorm
        
        # use only second half, since we only need t>k
        acor = acor[int(len(acor)/2):]
                
        return acor

    def power_spectrum(ts, time_step): 
        
        #Calculate fft
        fft_ts = np.fft.rfft(ts)
        
        #power spectrum is the square of magnitude of fft
        ps = np.abs(fft_ts)**2
        
        #fetching frequencies
        freqs = np.fft.rfftfreq(ts.size, time_step)
        idx = np.argsort(freqs)
        
        return freqs, ps
    
    acor = autocorrelation(phi)
    freqs, ps = power_spectrum(phi, time_step)

    # # plot the autocorrelation
    # # plt.figure().set_size_inches(12,4)
    # plt.plot(acor,label='phi')
    # # plt.xlim(0,1000)
    # plt.xlabel(r'$\tau$'+' (delay)')
    # plt.ylabel('Autocorrelation')
    # plt.legend()
    # plt.show()

    # plotting the power spectrum
    # plt.figure().set_size_inches(15,4)
    plt.plot(freqs, ps)
    # plt.axvline(x = 0.025, color = 'r', linestyle = 'dashed')
    # plt.axvline(x = 0.5, color = 'r', linestyle = 'dashed')
    plt.axvline(x = 1.0/np.pi, color = 'r', linestyle = 'dashed')
    # plt.axvline(x = 0.1, color = 'r', linestyle = 'dashed')
    plt.xlim(xlim[0], xlim[1]) #zoom in or out based on data
    plt.yscale('log')
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (log scale)")
    plt.title("Spectral density (log scale)")
    plt.show()

    # plotting the power spectrum
    # plt.figure().set_size_inches(15,4)
    plt.plot(freqs, ps)
    # plt.axvline(x = 0.025, color = 'r', linestyle = 'dashed')
    # plt.axvline(x = 0.5, color = 'r', linestyle = 'dashed')
    # plt.axvline(x = 0.2, color = 'r', linestyle = 'dashed')
    plt.axvline(x = 1.0/np.pi, color = 'r', linestyle = 'dashed')
    plt.xlim(xlim[0], xlim[1]) #zoom in or out based on data
    plt.ylim(0.0,1.0e8) #zoom in or out based on data
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title("Spectral density")
    plt.legend()
    plt.show()

    # # plotting the power spectrum
    # # plt.figure().set_size_inches(15,4)
    # plt.plot(np.log(freqs), ps)
    # # plt.xlim(0,4) #zoom in or out based on data
    # plt.grid()
    # plt.xlabel("log of Frequency")
    # plt.ylabel("Power Spectral Density")
    # plt.legend()
    # plt.show()

    return None


def recurrence_plot(Rx_s, Ry_s, Rz_s, ts, tf, max_time, norm_type, eps, n_step, n_bins):

    ts = ts[ts < max_time]
    num_times = len(ts)
    x_pts = []
    y_pts = []
    all_diffs = []
    for i in range(0, num_times, n_step):
        current_y = []
        for j in range(i, num_times, n_step):
            v = np.array([Rx_s[i] - Rx_s[j], Ry_s[i] - Ry_s[j], Rz_s[i] - Rz_s[j]])
            if norm_type == 'euclidean':
                dist = norm(v, ord=2)
            elif norm_type == 'max':
                dist = norm(v, ord=np.inf)
            else:
                print("unknown norm")
            if dist < (eps):
                x_pts.append(ts[i])
                y_pts.append(ts[j])
                if i != j:
                    x_pts.append(ts[j])
                    y_pts.append(ts[i])
                for k in range(len(current_y)):
                    diff = ts[j] - current_y[k]
                    if diff > 2*eps:
                        all_diffs.append(diff)
                current_y.append(ts[j])

    # fig, ax=plt.subplots()
    plt.xlim((0, 0.97*max_time))
    plt.ylim((0, 0.97*max_time))
    plt.plot(x_pts, y_pts, 'bo', markersize=1)
    plt.xlabel('t_n')
    plt.ylabel('t_n+1')
    plt.title('Recurrence plot')
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    # ax.axvline(x=2*np.pi, color = 'y', linestyle = ':', label='2_pi_v')
    # ax.axvline(x=4*np.pi, color = 'k', linestyle = ':', label='4_pi_v')
    plt.legend()
    plt.show()

    # Histogram of time differences
    # all_diffs = np.around(all_diffs)
    # all_diffs = all_diffs[all_diffs > 0]
    plt.xlim((0, 0.9*max_time))
    plt.hist(all_diffs, bins=n_bins)
    # plt.axvline(x = 4.0*np.pi, color = 'r', linestyle = 'dashed')
    plt.xlabel('Time interval')
    plt.ylabel('Count')
    plt.title('Historgram of spacing counts between recurrence lines')
    plt.show()

    return None

def strobe_map_many(t0, tf, w_version, w_params, t_eval, t_step_max, use_equal_times, T, proj_2d):

    Rx_all = [math.pi/8.0, math.pi/4.0, math.pi/2.0, math.pi, 0.0, 0.0, 0.0, 0.0, math.pi/8.0, math.pi/4.0, math.pi/2.0, math.pi]
    Ry_all = [0.0, 0.0, 0.0, 0.0, math.pi/8.0, math.pi/4.0, math.pi/2.0, math.pi, math.pi/8.0, math.pi/4.0, math.pi/2.0, math.pi]
    Rz_all = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    num_ics = 12

    for j in range(num_ics):
        Rx_0 = Rx_all[j]
        Ry_0 = Ry_all[j]
        Rz_0 = Rz_all[j]

        if use_equal_times:
            vs = solve_ivp(f_1, t_span = (t0, tf), y0 = [Rx_0, Ry_0, Rz_0], method='RK45', t_eval=t_eval, args=(w_version, w_params))
        else:
            vs = solve_ivp(f_1, t_span = (t0, tf), y0 = [Rx_0, Ry_0, Rz_0], method='RK45', max_step=t_step_max, args=(w_version, w_params))

        Rx_s = vs.y[0]
        Ry_s = vs.y[1]
        Rz_s = vs.y[2]
        ts = vs.t

        pi_mult = 0
        t_last = 0.0
        mask = np.array(len(ts)*[False])
        for i in range(len(ts)):
            t = ts[i]
            if (t >= pi_mult*T) and (t_last <= pi_mult*T):
                mask[i] = True
                pi_mult += 1
            t_last = t

        Rx_out = Rx_s[mask]
        Ry_out = Ry_s[mask]
        Rz_out = Rz_s[mask]
        t_out = ts[mask]

        if proj_2d == True: 
            # 2D projection
            if j == 0:
                fig, ax=plt.subplots()
            plt.plot(Rx_out, Ry_out, 'bo', markersize=2)
            plt.xlim((-7, 7))
            plt.ylim((-7, 7))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('period T time strobe soln projection')
            plt.legend()

        else:
            # 3D
            if j == 0:
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
            plt.gcf().set_size_inches(11.7,8.3) 
            ax.plot(Rx_out, Ry_out, Rz_out, 'bo', markersize=2)
            ax.set_xlim((-7.0, 7.0))
            ax.set_ylim((-7.0, 7.0)) 
            ax.set_zlim((-7.0, 7.0)) 
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('period T time strobe plot')
            ax.legend()

    plt.show()

    return None

def poincare_strobe(Rx_s, Ry_s, Rz_s, ts, T):
   
    pi_mult = 0
    t_last = 0.0
    mask = np.array(len(ts)*[False])
    for i in range(len(ts)):
        t = ts[i]
        if (t >= pi_mult*T) and (t_last <= pi_mult*T):
            mask[i] = True
            pi_mult += 1
        t_last = t

    Rx_out = Rx_s[mask]
    Ry_out = Ry_s[mask]
    Rz_out = Rz_s[mask]
    t_out = ts[mask]

    # 3D strobe plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    plt.gcf().set_size_inches(11.7,8.3) 
    ax.plot(Rx_out, Ry_out, Rz_out, 'bo', markersize=2, label='period T time strobe soln')
    ax.set_xlim((-7.0, 7.0))
    ax.set_ylim((-7.0, 7.0)) 
    ax.set_zlim((-7.0, 7.0)) 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('period T time strobe plot')
    ax.legend()
    plt.show()

    # 2D projection
    fig, ax=plt.subplots()
    plt.plot(Rx_out, Ry_out, 'bo', markersize=2)
    plt.xlim((-7, 7))
    plt.ylim((-7, 7))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('period T time strobe soln projection')
    plt.legend()
    plt.show()

    # Return map
    phi_map = np.sqrt(Rx_out*Rx_out + Ry_out*Ry_out + Rz_out*Rz_out)

    fig, ax=plt.subplots()
    plt.xlim((-1, 7))
    plt.ylim((-1, 7))
    ax.axline([0, 0], [1, 1])
    plt.plot(phi_map[:-1], phi_map[1:], 'ro', markersize=2, alpha=0.5)
    plt.xlabel('phi_n')
    plt.ylabel('phi_n+1')
    plt.title('Return map for period T strobe plot')
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    # ax.axvline(x=2*np.pi, color = 'y', linestyle = ':', label='2_pi_v')
    # ax.axvline(x=4*np.pi, color = 'k', linestyle = ':', label='4_pi_v')
    plt.legend()
    plt.show()

    return Rx_out, Ry_out, Rz_out, t_out

def zero_one_test(x):

    N = len(x)
    c = (7.0*np.pi)/8

    cos_vec = np.zeros(N)
    sin_vec = np.zeros(N)
    p = np.zeros(N)
    q = np.zeros(N)

    for i in range(N):
        cos_vec[i] = np.cos((i+1)*c) 
        sin_vec[i] = np.sin((i+1)*c) 

    for n in range(N):
        x_a = x[:(n+1)]
        cos_vec_a = cos_vec[:(n+1)]
        sin_vec_a = sin_vec[:(n+1)]
        p[n] = np.dot(x_a, cos_vec_a)
        q[n] = np.dot(x_a, sin_vec_a)

    n_cut = int(N/(10.0))
    M = np.zeros(n_cut)
    for n in range(n_cut):
        term = 0.0
        for j in range(n_cut):
            term += (p[j+n+1] - p[j])**2 + (q[j+n+1] - q[j])**2
        term = term/n_cut
        M[n] = term

    fig, ax=plt.subplots()
    plt.xlim((-1.3, 1.3))
    plt.ylim((-1.3, 1.3))
    plt.plot(p, q)
    plt.xlabel('p')
    plt.ylabel('q')
    # plt.title(title)
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    # ax.axvline(x=2*np.pi, color = 'y', linestyle = ':', label='2_pi_v')
    # ax.axvline(x=4*np.pi, color = 'k', linestyle = ':', label='4_pi_v')
    # plt.legend()
    plt.show()

    fig, ax=plt.subplots()
    n_vals = np.array(range(n_cut))
    plt.xlim((0, n_vals[-1]))
    plt.ylim((0, 1.1*np.max(M)))

    plt.plot(n_vals, M)
    plt.xlabel('n')
    plt.ylabel('M')
    # plt.title(title)
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    # ax.axvline(x=2*np.pi, color = 'y', linestyle = ':', label='2_pi_v')
    # ax.axvline(x=4*np.pi, color = 'k', linestyle = ':', label='4_pi_v')
    # plt.legend()
    plt.show()


    return None

def lyapunov_exponent_savary(Rx_0, Ry_0, Rz_0, t0, t_step_max, averaging, n_steps):
    from lyapynov import mLCE, LCE, CLV, ADJ

    # See  https://github.com/ThomasSavary08/Lyapynov and referenced paper for theory

    Euler_sys = ContinuousDS(np.array([Rx_0, Ry_0, Rz_0]), t0, f_ly, jacobian_ly, t_step_max)

    # # Compute mLCE (maximum Lyapunov Characteristic Exponent)
    # mLCE, history = mLCE(Euler_sys, 0, n_steps, True)

    # # Plot of mLCE evolution
    # plt.figure(figsize = (10,6))
    # plt.plot(history[:500000])
    # plt.axhline(y = 0.0, color = 'r', linestyle = '-') 
    # plt.xlabel("Number of time steps")
    # plt.ylabel("max lyapunov exponent")
    # plt.title("Evolution of the max Lyapunov Exponent")
    # plt.show()

    # Computation of LCE (Lyapunov Characteristic Exponents)
    LCE, history = LCE(Euler_sys, 3, 0, n_steps, keep=True)

    # Plot of LCE
    plt.figure(figsize = (10,6))
    plt.plot(history)
    plt.axhline(y = 0.0, color = 'k', linestyle = '-') 
    plt.xlabel("Number of time steps")
    plt.ylabel("Lyapunov Exponents")
    plt.title("Evolution of the Lyapunov Exponents")
    plt.show()

    if averaging:
        # Compute CLV
        CLV, traj, checking_ds = CLV(Euler_sys, 3, 0, n_steps, n_steps, n_steps, True, check = True)

        # Check CLV
        LCE_check = np.zeros((Euler_sys.dim,))
        for i in range(len(CLV)):
            W = CLV[i]
            init_norm = np.linalg.norm(W, axis = 0)
            W = checking_ds.next_LTM(W)
            norm = np.linalg.norm(W, axis = 0)
            checking_ds.forward(1, False)
            LCE_check += np.log(norm / init_norm) / checking_ds.dt
        LCE_check = LCE_check / len(CLV)
        print(LCE_check)

        # print("Average of first local Lyapunov exponent: {:.3f}".format(LCE_check[0]))
        # print("Average of second local Lyapunov exponent: {:.3f}".format(LCE_check[1]))
        # print("Average of third local Lyapunov exponent: {:.3f}".format(LCE_check[2]))
        return LCE_check
    else:
        return LCE
    

def lyapunov_exponent(Rx_0, Ry_0, Rz_0, t0, tf, t_step_max, t_start, delt, w_version, w_params, use_equal_times):

    if use_equal_times:
        t_eval = np.arange(t0, tf, t_step_max)  # If using equally spaced predetermined times
        vs0 = solve_ivp(f_1, t_span = (t0, tf), y0 = [Rx_0, Ry_0, Rz_0], method='RK45', t_eval=t_eval, args=(w_version, w_params))
    else:
        vs0 = solve_ivp(f_1, t_span = (t0, tf), y0 = [Rx_0, Ry_0, Rz_0], method='RK45', max_step=t_step_max, args=(w_version, w_params))
    Rx_s0 = vs0.y[0]
    Ry_s0 = vs0.y[1]
    Rz_s0 = vs0.y[2]
    ts0 = vs0.t

    Rx_1 = Rx_0 + delt
    Ry_1 = Ry_0 + delt 
    Rz_1 = Rz_0 + delt 

    # Initial separation distance
    dR_0 = np.sqrt((Rx_1 - Rx_0)**2 + (Ry_1 - Ry_0)**2 + (Rz_1 - Rz_0)**2)

    if use_equal_times:
        t_eval = np.arange(t0, tf, t_step_max)  # If using equally spaced predetermined times
        vs1 = solve_ivp(f_1, t_span = (t0, tf), y0 = [Rx_1, Ry_1, Rz_1], method='RK45', t_eval=ts0, args=(w_version, w_params))
    else:
        vs1 = solve_ivp(f_1, t_span = (t0, tf), y0 = [Rx_1, Ry_1, Rz_1], method='RK45', max_step=t_step_max, args=(w_version, w_params))
    Rx_s1 = vs1.y[0]
    Ry_s1 = vs1.y[1]
    Rz_s1 = vs1.y[2]
    ts1 = vs1.t

    # # Final separation dist
    dR_f = np.sqrt((Rx_s1[-1] - Rx_s0[-1])**2 + (Ry_s1[-1] - Ry_s0[-1])**2 + (Rz_s1[-1] - Rz_s0[-1])**2 ) 
    # Final Lyap estimate
    ly_est = (np.log((dR_f/dR_0)))/tf

    tc = ts0[:]
    dR = np.zeros(len(tc))
    ly_all = np.zeros(len(tc))
    for i, t_i in enumerate(tc):
        dR[i] = np.sqrt((Rx_s1[i] - Rx_s0[i])**2 + (Ry_s1[i] - Ry_s0[i])**2 + (Rz_s1[i] - Rz_s0[i])**2 ) 
        ly_all[i] = (np.log((dR[i]/dR_0)))/t_i

    # Compare nearby starting conditions in time series plots to check for divergence of trajectories
    plot_single_var_vs_time(ts0, Rx_s0, tlim=(0, tf), xlim=(-8, 8), xlabel='t', ylabel='Ex', title="Ex versus t")
    plot_single_var_vs_time(ts0, Rz_s0, tlim=(0, tf), xlim=(-8, 8), xlabel='t', ylabel='Ez', title="Ez versus t")
    plot_single_var_vs_time(ts1, Rx_s1, tlim=(0, tf), xlim=(-8, 8), xlabel='t', ylabel='Ex_p', title="perturbed IC of Ex versus t")
    plot_single_var_vs_time(ts1, Rz_s1, tlim=(0, tf), xlim=(-8, 8), xlabel='t', ylabel='Ez_p', title="perturbed IC of Ez versus t")

    # Plot lyap
    fig, ax=plt.subplots()
    plt.xlim([t_start, tf])
    plt.ylim([-0.1, 0.1])
    plt.plot(tc[t_start:], ly_all[t_start:], label='Lyap exp')
    plt.xlabel('time')
    plt.ylabel('Lyap exp')
    plt.title('Lyapunov over time')
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    # ax.axvline(x=2*np.pi, color = 'y', linestyle = ':', label='2_pi_v')
    # ax.axvline(x=4*np.pi, color = 'k', linestyle = ':', label='4_pi_v')
    plt.legend()
    plt.show()

    print(ly_est)

    return ly_est


def map_successive_max(v, var_name):

    v_max = np.array([])

    for j in range(2, len(v)):
        if (v[j] < v[j-1]) & (v[j-1] > v[j-2]):
            v_max = np.append(v_max, v[j-1])

    fig, ax=plt.subplots()
    plt.xlim((-1, 7))
    plt.ylim((-1, 7))
    ax.axline([0, 0], [1, 1])
    plt.plot(v_max[:-1], v_max[1:], 'ro', markersize=2, alpha=0.5)
    plt.xlabel(var_name + '_n')
    plt.ylabel(var_name + '_n+1')
    plt.title('Return map for maxima of ' + var_name)
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    # ax.axvline(x=2*np.pi, color = 'y', linestyle = ':', label='2_pi_v')
    # ax.axvline(x=4*np.pi, color = 'k', linestyle = ':', label='4_pi_v')
    plt.legend()
    plt.show()

    return None

def poincare_section(Rx_s, Ry_s, Rz_s, ts, normalized_lengths):
    # Components of Normal vector to plane
    Nx = 1
    Ny = 0
    Nz = 0
    # Point on the plane
    Px = 0
    Py = 0
    Pz = 0

    if normalized_lengths:
        len_R = np.sqrt(Rx_s*Rx_s + Ry_s*Ry_s + Rz_s*Rz_s)
        Rx_s = Rx_s/len_R
        Rx_s = Ry_s/len_R
        Rx_s = Rz_s/len_R

    U_last = -1
    y_map = np.array([])
    z_map = np.array([])
    t_map = np.array([])
    for i in range(len(ts)):
        # Plane function
        U = Nx*(Rx_s[i] - Px) + Ny*(Ry_s[i] - Py) + Nz*(Rz_s[i] - Pz)
        if (U < 0) & (U_last > 0):
            Rx_a = Rx_s[i-1]
            Rx_b = Rx_s[i]
            Rx_interp = (Rx_a + Rx_b)/2.0
            Ry_a = Ry_s[i-1]
            Ry_b = Ry_s[i]
            Ry_interp = (Ry_a + Ry_b)/2.0
            Rz_a = Rz_s[i-1]
            Rz_b = Rz_s[i]
            Rz_interp = (Rz_a + Rz_b)/2.0
            t_a = ts[i-1]
            t_b = ts[i]
            t_interp = (t_a + t_b)/2.0

            vp = solve_ivp(f_p, t_span = (Rx_a, 0), y0 = [t_a, Ry_a, Rz_a], method='RK45', max_step = 0.01, args=(w_version, w_params))
            t_vp_all = vp.y[0]
            Ry_vp_all = vp.y[1]
            Rz_vp_all = vp.y[2]
            Rx_vp_all = vp.t
            t_vp = t_vp_all[-1]
            Ry_vp = Ry_vp_all[-1]
            Rz_vp = Rz_vp_all[-1]

            y_map = np.append(y_map, Ry_vp)
            z_map = np.append(z_map, Rz_vp)
            t_map = np.append(t_map, t_vp)
            t_diff = np.diff(t_map)

            # y_map = np.append(y_map, Ry_interp)
            # z_map = np.append(z_map, Rz_interp)
            # t_map = np.append(t_map, t_interp)

        U_last = U

    fig, ax=plt.subplots()
    plt.xlim((-7, 7))
    plt.ylim((-7, 7))
    plt.plot(y_map, z_map, 'ro', markersize=2, alpha=0.5, label='Poincare x=0')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Poincare Section x = 0')
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    # ax.axvline(x=2*np.pi, color = 'y', linestyle = ':', label='2_pi_v')
    # ax.axvline(x=4*np.pi, color = 'k', linestyle = ':', label='4_pi_v')
    plt.legend()
    plt.show()

    phi_map = np.sqrt(y_map*y_map + z_map*z_map)
  
    fig, ax=plt.subplots()
    plt.xlim((0, 7))
    plt.ylim((0, 7))
    ax.axline([0, 0], [1, 1])
    plt.plot(phi_map[:-1], phi_map[1:], 'ro', markersize=2, alpha=0.5)
    plt.xlabel('phi_n')
    plt.ylabel('phi_n+1')
    plt.title('Return map for Poincare section at x=0')
    # ax.axhline(y=2*np.pi, color = 'red', linestyle = ':', label='2_pi_h')
    # ax.axvline(x=np.pi, color = 'm', linestyle = ':', label='pi_v')
    # ax.axvline(x=2*np.pi, color = 'y', linestyle = ':', label='2_pi_v')
    # ax.axvline(x=4*np.pi, color = 'k', linestyle = ':', label='4_pi_v')
    plt.legend()
    plt.show()

    return y_map, z_map, t_map

# Enter differential equation
def f_p(Rx, Y, w_version, w_params):
    t, Ry, Rz = Y
    wx, wy, wz = get_angular_velocity(t, np.array([Rx, Ry, Rz]), w_version, w_params)
    len_R_sq = Rx*Rx + Ry*Ry + Rz*Rz
    len_R = np.sqrt(len_R_sq)

    g = eval_auxiliary_func(len_R)

    R_dot_w = Rx*wx + Ry*wy + Rz*wz

    if len_R_sq == 0.0:
        dx = g*wx 
        dy = g*wy  
        dz = g*wz 
    elif len_R > 1.5*np.pi:
        dx = (wx - Rx*(R_dot_w / len_R_sq))*g + (R_dot_w / len_R_sq)*Rx + 0.5*(wy*Rz - wz*Ry)  # formula for dx/dt
        dy = (wy - Ry*(R_dot_w / len_R_sq))*g + (R_dot_w / len_R_sq)*Ry + 0.5*(wz*Rx - wx*Rz)  # formula for dy/dt
        dz = (wz - Rz*(R_dot_w / len_R_sq))*g + (R_dot_w / len_R_sq)*Rz + 0.5*(wx*Ry - wy*Rx)  # formula for dz/dt
    else:
        dx = g*wx + (R_dot_w / len_R_sq ) * Rx * (1-g) + 0.5*(wy*Rz - wz*Ry)  # formula for dx/dt
        dy = g*wy + (R_dot_w / len_R_sq ) * Ry * (1-g) + 0.5*(wz*Rx - wx*Rz)  # formula for dy/dt
        dz = g*wz + (R_dot_w / len_R_sq ) * Rz * (1-g) + 0.5*(wx*Ry - wy*Rx)  # formula for dz/dt

    d1 = 1.0/dx
    d2 = dy/dx
    d3 = dz/dx

    return [d1, d2, d3]  


def jacobian(t, Y, w_version, w_params, return_eigen=False):
    Rx, Ry, Rz = Y
    wx, wy, wz = get_angular_velocity(t, Y, w_version, w_params)

    version = 1

    if version == 1:
        dot = wx*Rx + wy*Ry + wz*Rz

        J_xx = dot + Rx*wx
        J_xy = Rx*wy - 2.0*wz
        J_xz = Rx*wz + 2.0*wy

        J_yx = Ry*wx + 2.0*wz
        J_yy = dot + Ry*wy
        J_yz = Ry*wz - 2*wx

        J_zx = Rz*wx - 2.0*wy
        J_zy = Rz*wy + 2.0*wx
        J_zz = dot + Rz*wz

        J = np.array([[J_xx, J_xy, J_xz],[J_yx, J_yy, J_yz],[J_zx, J_zy, J_zz]])    
        J = 0.25*J   
    
    elif version == 2:
        k = -1.0/(2.0*np.sqrt(4 - (Rx*Rx + Ry*Ry + Rz*Rz)))

        J_xx = k*Rx*wx
        J_xy = k*Ry*wx - 0.5*wz
        J_xz = k*Rz*wx + 0.5*wy

        J_yx = k*Rx*wy + 0.5*wz
        J_yy = k*Ry*wy
        J_yz = k*Rz*wy - 0.5*wx

        J_zx = k*Rx*wz - 0.5*wy
        J_zy = k*Ry*wz + 0.5*wx
        J_zz = k*Rz*wz

        J = np.array([[J_xx, J_xy, J_xz],[J_yx, J_yy, J_yz],[J_zx, J_zy, J_zz]])

    if return_eigen:
        evals, evecs = linalg.eig(J)  
        return J, evals, evecs
    else:
        return J
    
def jacobian_ly(Y, t):
    return_eigen=False
    # Rx, Ry, Rz = Y
    Rx = Y[0]
    Ry = Y[1]
    Rz = Y[2]
    Y = None
    wx, wy, wz = get_angular_velocity(t, Y, w_version, w_params)

    version = 3

    if version == 1:
        dot = wx*Rx + wy*Ry + wz*Rz

        J_xx = dot + Rx*wx
        J_xy = Rx*wy - 2.0*wz
        J_xz = Rx*wz + 2.0*wy

        J_yx = Ry*wx + 2.0*wz
        J_yy = dot + Ry*wy
        J_yz = Ry*wz - 2.0*wx

        J_zx = Rz*wx - 2.0*wy
        J_zy = Rz*wy + 2.0*wx
        J_zz = dot + Rz*wz

        J = np.array([[J_xx, J_xy, J_xz],[J_yx, J_yy, J_yz],[J_zx, J_zy, J_zz]])    
        J = 0.25*J   
    
    elif version == 2:
        k = -1.0/(2.0*np.sqrt(4 - (Rx*Rx + Ry*Ry + Rz*Rz)))

        J_xx = k*Rx*wx
        J_xy = k*Ry*wx - 0.5*wz
        J_xz = k*Rz*wx + 0.5*wy

        J_yx = k*Rx*wy + 0.5*wz
        J_yy = k*Ry*wy
        J_yz = k*Rz*wy - 0.5*wx

        J_zx = k*Rx*wz - 0.5*wy
        J_zy = k*Ry*wz + 0.5*wx
        J_zz = k*Rz*wz

        J = np.array([[J_xx, J_xy, J_xz],[J_yx, J_yy, J_yz],[J_zx, J_zy, J_zz]])

    elif version == 3:
 
        R_len = np.sqrt(Rx*Rx + Ry*Ry + Rz*Rz)
        dot_prod = Rx*wx + Ry*wy + Rz*wz

        if R_len > 0.1:
            g_frac= -(1.0 - (np.sin(R_len)/R_len))/(2.0 - 2.0*np.cos(R_len))
            h_frac = (R_len + np.sin(R_len))/(2*(R_len**3)*(1.0 - np.cos(R_len))) - 2.0/(R_len**4)
            h = 1.0/(R_len*R_len) - np.sin(R_len)/(2*R_len*(1.0 - np.cos(R_len)))
        else:
            g_frac = -1.0/6.0 - (R_len*R_len)/180.0 - (R_len**4)/5040.0
            h_frac = 1.0/360.0 + (R_len*R_len)/7560.0 + (R_len**4)/201600.0
            h = 1.0/12.0 + (R_len*R_len)/720.0 + (R_len**4)/30240.0

        w_xt = np.array([np.array([wx*Rx, wx*Ry, wx*Rz]), np.array([wy*Rx, wy*Ry, wy*Rz]), np.array([wz*Rx, wz*Ry, wz*Rz])])

        x_wt = np.array([np.array([wx*Rx, wy*Rx, wz*Rx]), np.array([wx*Ry, wy*Ry, wz*Ry]), np.array([wx*Rz, wy*Rz, wz*Rz])])

        x_xt = np.array([np.array([Rx*Rx, Ry*Rx, Rz*Rx]), np.array([Rx*Ry, Ry*Ry, Rz*Ry]), np.array([Rx*Rz, Ry*Rz, Rz*Rz])])

        I_mat = np.identity(3)

        omg_mat = 0.5*np.array([np.array([0.0, -wz, wy]), np.array([wz, 0.0, -wx]), np.array([-wy, wx, 0.0])])

        J = g_frac*w_xt + h*(dot_prod*I_mat + x_wt) + h_frac*dot_prod*x_xt + omg_mat
        

    if return_eigen:
        evals, evecs = linalg.eig(J)  
        return J, evals, evecs
    else:
        return J
    
# Enter differential equation for Lyap calc
def f_ly(Y, t):
    # Rx, Ry, Rz = Y
    Rx = Y[0]
    Ry = Y[1]
    Rz = Y[2]
    Y = None
    wx, wy, wz = get_angular_velocity(t, Y, w_version, w_params)
    len_R_sq = Rx*Rx + Ry*Ry + Rz*Rz
    len_R = np.sqrt(len_R_sq)

    g = eval_auxiliary_func(len_R)

    R_dot_w = Rx*wx + Ry*wy + Rz*wz

    if len_R_sq == 0.0:
        dx = g*wx 
        dy = g*wy  
        dz = g*wz 
    elif len_R > 1.5*np.pi:
        dx = (wx - Rx*(R_dot_w / len_R_sq))*g + (R_dot_w / len_R_sq)*Rx + 0.5*(wy*Rz - wz*Ry)  # formula for dx/dt
        dy = (wy - Ry*(R_dot_w / len_R_sq))*g + (R_dot_w / len_R_sq)*Ry + 0.5*(wz*Rx - wx*Rz)  # formula for dy/dt
        dz = (wz - Rz*(R_dot_w / len_R_sq))*g + (R_dot_w / len_R_sq)*Rz + 0.5*(wx*Ry - wy*Rx)  # formula for dz/dt
    else:
        dx = g*wx + (R_dot_w / len_R_sq ) * Rx * (1-g) + 0.5*(wy*Rz - wz*Ry)  # formula for dx/dt
        dy = g*wy + (R_dot_w / len_R_sq ) * Ry * (1-g) + 0.5*(wz*Rx - wx*Rz)  # formula for dy/dt
        dz = g*wz + (R_dot_w / len_R_sq ) * Rz * (1-g) + 0.5*(wx*Ry - wy*Rx)  # formula for dz/dt

    return np.array([dx, dy, dz])  


def f_ly_2(Y, t):
    x = Y[0]
    y = Y[1]
    z = Y[2]
    Y = None

    wx, wy, wz = get_angular_velocity(t, Y, w_version, w_params)

    R_dot_w = x*wx + y*wy + z*wz

    dx = wx + 0.25*x*R_dot_w + 0.5*(wy*z - wz*y)
    dy = wy + 0.25*y*R_dot_w + 0.5*(wz*x - wx*z)
    dz = wz + 0.25*z*R_dot_w + 0.5*(wx*y - wy*x)

    return np.array([dx, dy, dz])  

# Enter differential equation
def f_gibbs(t, Y, w_version, w_params):
    x, y, z = Y

    wx, wy, wz = get_angular_velocity(t, Y, w_version, w_params)

    R_dot_w = x*wx + y*wy + z*wz

    dx = wx + 0.25*x*R_dot_w + 0.5*(wy*z - wz*y)
    dy = wy + 0.25*y*R_dot_w + 0.5*(wz*x - wx*z)
    dz = wz + 0.25*z*R_dot_w + 0.5*(wx*y - wy*x)

    return [dx, dy, dz]  

# Enter differential equation
def f_1(t, Y, w_version, w_params):
    Rx, Ry, Rz = Y
    wx, wy, wz = get_angular_velocity(t, Y, w_version, w_params)
    len_R_sq = Rx*Rx + Ry*Ry + Rz*Rz
    len_R = np.sqrt(len_R_sq)

    g = eval_auxiliary_func(len_R)

    R_dot_w = Rx*wx + Ry*wy + Rz*wz

    if len_R_sq == 0.0:
        dx = g*wx 
        dy = g*wy  
        dz = g*wz 
    elif len_R > 1.5*np.pi:
        dx = (wx - Rx*(R_dot_w / len_R_sq))*g + (R_dot_w / len_R_sq)*Rx + 0.5*(wy*Rz - wz*Ry)  # formula for dx/dt
        dy = (wy - Ry*(R_dot_w / len_R_sq))*g + (R_dot_w / len_R_sq)*Ry + 0.5*(wz*Rx - wx*Rz)  # formula for dy/dt
        dz = (wz - Rz*(R_dot_w / len_R_sq))*g + (R_dot_w / len_R_sq)*Rz + 0.5*(wx*Ry - wy*Rx)  # formula for dz/dt
    else:
        dx = g*wx + (R_dot_w / len_R_sq ) * Rx * (1-g) + 0.5*(wy*Rz - wz*Ry)  # formula for dx/dt
        dy = g*wy + (R_dot_w / len_R_sq ) * Ry * (1-g) + 0.5*(wz*Rx - wx*Rz)  # formula for dy/dt
        dz = g*wz + (R_dot_w / len_R_sq ) * Rz * (1-g) + 0.5*(wx*Ry - wy*Rx)  # formula for dz/dt

    return [dx, dy, dz]  

# Enter differential equation
def f_2(t, Y):
    global phi_increasing
    global phi_glob

    Rx, Ry, Rz = Y
    wx, wy, wz = get_angular_velocity(t)
    len_R_sq = Rx*Rx + Ry*Ry + Rz*Rz
    len_R = np.sqrt(len_R_sq)
    w_dot_R = wx*Rx + wy*Ry + wz*Rz

    eps = 0.00001

    if (4.0 - len_R_sq < eps):
        if w_dot_R > 0:
            phi_increasing = -1.0 
        else:
            phi_increasing = 1.0
    
    if phi_increasing == 1.0:
        phi = 2*np.arcsin(len_R/2.0)
    elif phi_increasing == -1.0:
        phi = 2*np.pi - 2*np.arcsin(len_R/2.0)

    phi_glob = np.append(phi_glob, [[phi, t]], axis=0)
       
    cos_term = phi_increasing * (0.5 * np.sqrt(4 - len_R_sq))  # for f = 2*sin(phi/2)

    dx = cos_term*wx + 0.5*(wy*Rz - wz*Ry)  # formula for dx/dt
    dy = cos_term*wy + 0.5*(wz*Rx - wx*Rz)  # formula for dy/dt
    dz = cos_term*wz + 0.5*(wx*Ry - wy*Rx)   # formula for dz/dt

    return [dx, dy, dz]  

# Enter differential equation
def f_3(t, Y, w_version, w_params):
    Rx, Ry, Rz, Rq = Y
    Y2 = Rx, Ry, Rz
    wx, wy, wz = get_angular_velocity(t, Y2, w_version, w_params)

    R_dot_w = Rx*wx + Ry*wy + Rz*wz

    dx = 0.5*Rq*wx + 0.5*(wy*Rz - wz*Ry)
    dy = 0.5*Rq*wy + 0.5*(wz*Rx - wx*Rz)
    dz = 0.5*Rq*wz + 0.5*(wx*Ry - wy*Rx)
    dq = -0.5*R_dot_w

    return [dx, dy, dz, dq]  


def f_b(t, Y, w_version, w_params):
    Bx, By, Bz = Y
    wx, wy, wz = get_angular_velocity(t, Y, w_version, w_params)

    dx = wy*Bz - By*wz
    dy = wz*Bx - Bz*wx
    dz = wx*By - Bx*wy

    return [dx, dy, dz]  

def f_lorenz(t, Y):

    x, y, z = Y
  
    dx = sig*(y - x)
    dy = x*(rho - z) - y
    dz = x*y - beta*z
  
    return [dx, dy, dz]  

def f_lorenz_ly(Y, t):
    
    x = Y[0]
    y = Y[1]
    z = Y[2]

    dx = sig*(y - x)
    dy = x*(rho - z) - y
    dz = x*y - beta*z
  
    return np.array([dx, dy, dz])  

def jac_lorenz_ly(Y, t):
   
    x = Y[0]
    y = Y[1]
    z = Y[2]
 
    J_xx = -sig
    J_xy = sig
    J_xz = 0.0

    J_yx = rho - z
    J_yy = -1.0
    J_yz = -x

    J_zx = y
    J_zy = x
    J_zz = -beta

    J = np.array([[J_xx, J_xy, J_xz],[J_yx, J_yy, J_yz],[J_zx, J_zy, J_zz]])

    return J

def f_henon(t, Y):

    px, py, x, y = Y
  
    dpx = -x - 2*x*y
    dpy = -y - x*x + y*y
    dx = px
    dy = py
  
    return [dpx, dpy, dx, dy]  

def f_henon_ly(Y, t):
    
    px = Y[0]
    py = Y[1]
    x = Y[2]
    y = Y[3]

    dpx = -x - 2*x*y
    dpy = -y - x*x + y*y
    dx = px
    dy = py
  
    return np.array([dpx, dpy, dx, dy])  

def jac_henon_ly(Y, t):
   
    px = Y[0]
    py = Y[1]
    x = Y[2]
    y = Y[3]
 
    J_11 = 0.0
    J_12 = 0.0
    J_13 = -1.0 - 2.0*y
    J_14 = -2.0*x

    J_21 = 0.0
    J_22 = 0.0
    J_23 = -2.0*x
    J_24 = -1.0 + 2.0*y

    J_31 = 1.0
    J_32 = 0.0
    J_33 = 0.0
    J_34 = 0.0

    J_41 = 0.0
    J_42 = 1.0
    J_43 = 0.0
    J_44 = 0.0

    J = np.array([[J_11, J_12, J_13, J_14],[J_21, J_22, J_23, J_24],[J_31, J_32, J_33, J_34], [J_41, J_42, J_43, J_44]])

    return J

def f_van(t, Y):

    x, y, z = Y
  
    dx = y - eps*((1.0/3.0)*(x**3) - x)
    dy = -x + F*np.cos(z)
    dz = omg
  
    return [dx, dy, dz]  

def f_van_ly(Y, t):
    
    x = Y[0]
    y = Y[1]
    z = Y[2]

    dx = y - eps*((1.0/3.0)*(x**3) - x)
    dy = -x + F*np.cos(z)
    dz = omg
  
    return np.array([dx, dy, dz])  

def jac_van_ly(Y, t):
   
    x = Y[0]
    y = Y[1]
    z = Y[2]
 
    J_xx = -eps*(x*x - 1.0)
    J_xy = 1.0
    J_xz = 0.0

    J_yx = -1.0
    J_yy = 0.0
    J_yz = -F*np.sin(z)

    J_zx = 0.0
    J_zy = 0.0
    J_zz = 0.0

    J = np.array([[J_xx, J_xy, J_xz],[J_yx, J_yy, J_yz],[J_zx, J_zy, J_zz]])

    return J

def test_systems(system, Y_0, t0, tf, t_step_max, params):
    from lyapynov import mLCE, LCE, CLV, ADJ

    if system == "van":
        x_0 = Y_0[0]
        y_0 = Y_0[1]
        z_0 = Y_0[2]

        vs = solve_ivp(f_van, t_span = (t0, tf), y0 = [x_0, y_0, z_0], method='RK45', max_step=t_step_max)

        x_s = vs.y[0]
        y_s = vs.y[1]
        z_s = vs.y[2]
        ts = vs.t

        x_dot_s = y_s - eps*((1.0/3.0)*(x_s**3) - x_s)

        plot_single_var_vs_time(ts, x_s, tlim=(0, tf), xlim=(-3, 3), xlabel='t', ylabel='x')

        fig, ax = plt.subplots(figsize=(11.7,8.3))
        ax.plot(x_s, x_dot_s, linewidth=2, label='norm ODE soln')  
        ax.legend()
        ax.plot([x_s[0]], [x_dot_s[0]], 'go', markersize=5, alpha=0.5)  
        ax.plot([x_s[-1]], [x_dot_s[-1]], 'ro', markersize=5, alpha=0.5)  
        # Label each axis
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # Set axis limits
        ax.set_xlim((-3, 3))
        ax.set_ylim((-3, 3))
        plt.show()

        vanderPol = ContinuousDS(np.array([x_0, y_0, z_0]), t0, f_van_ly, jac_van_ly, t_step_max)
        mLCE, history = mLCE(vanderPol, 0, 10**5, True)
        print(mLCE)
        LCE, history = LCE(vanderPol, 3, 0, 10**5, True)
        print(LCE)
        CLV, traj, checking_ds = CLV(vanderPol, 3, 0, 10**5, 10**5, 10**5, True, check = True)
        LCE_check = np.zeros((vanderPol.dim,))
        for i in range(len(CLV)):
            W = CLV[i]
            init_norm = np.linalg.norm(W, axis = 0)
            W = checking_ds.next_LTM(W)
            norm = np.linalg.norm(W, axis = 0)
            checking_ds.forward(1, False)
            LCE_check += np.log(norm / init_norm) / checking_ds.dt
        LCE_check = LCE_check / len(CLV)
        print(LCE_check)

    if system == "lorenz":

        x_0 = Y_0[0]
        y_0 = Y_0[1]
        z_0 = Y_0[2]

        vs = solve_ivp(f_lorenz, t_span = (t0, tf), y0 = [x_0, y_0, z_0], method='RK45', max_step=t_step_max)

        x_s = vs.y[0]
        y_s = vs.y[1]
        z_s = vs.y[2]
        ts = vs.t

        Lorenz63 = ContinuousDS(np.array([x_0, y_0, z_0]), t0, f_lorenz_ly, jac_lorenz_ly, t_step_max)
        mLCE, history = mLCE(Lorenz63, 0, 10**6, True)
        print(mLCE)
        LCE, history = LCE(Lorenz63, 3, 0, 10**6, True)
        print(LCE)
        CLV, traj, checking_ds = CLV(Lorenz63, 3, 0, 10**5, 10**6, 10**5, True, check = True)
        LCE_check = np.zeros((Lorenz63.dim,))
        for i in range(len(CLV)):
            W = CLV[i]
            init_norm = np.linalg.norm(W, axis = 0)
            W = checking_ds.next_LTM(W)
            norm = np.linalg.norm(W, axis = 0)
            checking_ds.forward(1, False)
            LCE_check += np.log(norm / init_norm) / checking_ds.dt
        LCE_check = LCE_check / len(CLV)
        print(LCE_check)

        lr_x = nolds.lyap_r(x_s, emb_dim=10, lag=None, min_tsep=None, tau=1, min_neighbors=20, trajectory_len=20, fit=u'RANSAC', debug_plot=False, debug_data=False, plot_file=None, fit_offset=0)
        lr_y = nolds.lyap_r(y_s, emb_dim=10, lag=None, min_tsep=None, tau=1, min_neighbors=20, trajectory_len=20, fit=u'RANSAC', debug_plot=False, debug_data=False, plot_file=None, fit_offset=0)
        lr_z = nolds.lyap_r(z_s, emb_dim=10, lag=None, min_tsep=None, tau=1, min_neighbors=20, trajectory_len=20, fit=u'RANSAC', debug_plot=False, debug_data=False, plot_file=None, fit_offset=0)

    if system == "henon":
        Henon = ContinuousDS(np.array([px_0, py_0, x_0, y_0]), t0, f_henon_ly, jac_henon_ly, t_step_max)
        mLCE, history = mLCE(Henon, 100, 10**5, True)
        print(mLCE)
        LCE, history = LCE(Henon, 4, 100, 10**5, True)
        print(LCE)
        CLV, traj, checking_ds = CLV(Henon, 4, 100, 10**5, 10**5, 10**5, True, check = True)
        LCE_check = np.zeros((Henon.dim,))
        for i in range(len(CLV)):
            W = CLV[i]
            init_norm = np.linalg.norm(W, axis = 0)
            W = checking_ds.next_LTM(W)
            norm = np.linalg.norm(W, axis = 0)
            checking_ds.forward(1, False)
            LCE_check += np.log(norm / init_norm) / checking_ds.dt
        LCE_check = LCE_check / len(CLV)
        print(LCE_check)


    return None


########################################################### Main ###############################################################################################################


# system = "lorenz"
# t0 = 0.0001
# tf = 100.0 
# t_step_max = 0.01
# x_0 = 1.5
# y_0 = -1.5
# z_0 = 20.0
# Y_0 = [x_0, y_0, z_0]
# sig = 16.0
# rho = 45.92
# beta = 4.0
# params = [sig, rho, beta]
# test_systems(system, Y_0, t0, tf, t_step_max, params)

# system = "van"
# t0 = 0.0001
# tf = 1000.0 
# t_step_max = 0.01
# x_0 = 0.5
# y_0 = 0.5
# z_0 = 0.0
# Y_0 = [x_0, y_0, z_0]
# F = 1.0
# eps = 0.5
# omg = 0.5
# params = [F, eps, omg]
# test_systems(system, Y_0, t0, tf, t_step_max, params)

########## GLOBALS ###########

# # For diff eq method 2
# global phi_increasing  
# phi_increasing = 1.0
# global phi_glob  
# phi_glob = np.empty(shape=[0, 2])

# # For impulse w
# global iter_num  
# iter_num = 1
# global wx_curr  
# wx_curr = 0.0
# global wy_curr
# wy_curr = 0.0
# global wz_curr
# wz_curr = 1.0

################################

# Enter range of time
t0 = 0.0001
# tf = 420.0
# tf = 32.0*np.pi
tf = 100.0
t_step_max = 0.01
use_equal_times = False
tfirst = None # first time for showing solution curve on plot
t_eval = np.arange(t0, tf, t_step_max)  # If using equally spaced predetermined times

# # Enter initial values for Euler vector
# Rx_0 = 1.0/np.sqrt(3) # initial value of x
# Ry_0 = 1.0/np.sqrt(3) # initial value of y
# Rz_0 = 1.0/np.sqrt(3) # initial value of z

# Rx_0 = 0.01 # initial value of x
# Ry_0 = 0.01 # initial value of y
# Rz_0 = 0.01 # initial value of z

Rx_0 = 1.0
Ry_0 = 0.0
Rz_0 = 0.0

# Initial Body orientation vector
Bx_0 = 0.0
By_0 = 1.0
Bz_0 = 0.0

diffeq_method = 1  # 1 or 2 or 3. 3 is quaternion version.

# w_version = "const_rotation"
w_version = "const_e3"
# w_version = "experiment_d_1"
# w_version = "torus_rotation"
# w_version = "back_forth_acc"
# w_version = "experiment_b_3"
# w_version = "experiment_a_1"

# # Have to define params based on specific w used
if (w_version == "const_rotation") | (w_version == "back_forth_acc"):
    T = 0.01*math.pi
    # T = 40.0  # whether or not period is multiple of pi doesn't change behavior, still appears quasiperiodic
    w1 = (2.0*np.pi)/T
    w_params = [w1]

elif (w_version == "torus_rotation") | (w_version == "experient_d_1"):
    T1 = 5.0
    T2 = 10.0
    w1 = (2.0*np.pi)/T1
    w2 = (2.0*np.pi)/T2
    r1 = 2.0
    r2 = 1.0
    w_params = [w1, w2, r1, r2]

elif w_version == "const_e3":
    T = 4.0*np.pi
    w1 = (2.0*np.pi)/T
    w_params = None

elif (w_version == "experiment_a_1") | (w_version == "experiment_a_2") | (w_version == "experiment_b_1") | (w_version == "experiment_b_2") | (w_version == "experiment_b_3"):
    cc = 1.0
    t_end = tf
    w1 = 1
    w_params = [cc, t_end, w1]

else:
    w_params = None

if diffeq_method == 1:
    if use_equal_times:
        vs = solve_ivp(f_1, t_span = (t0, tf), y0 = [Rx_0, Ry_0, Rz_0], method='RK45', t_eval=t_eval, args=(w_version, w_params))
    else:
        vs = solve_ivp(f_1, t_span = (t0, tf), y0 = [Rx_0, Ry_0, Rz_0], method='RK45', max_step=t_step_max, args=(w_version, w_params))
    
elif diffeq_method == 2:
    vs = solve_ivp(f_2, t_span = (t0, tf), y0 = [Rx_0, Ry_0, Rz_0], method='RK45', max_step=t_step_max, args=None)

elif diffeq_method == 3:
    len_R_0 = np.sqrt(Rx_0*Rx_0 + Ry_0*Ry_0 + Rz_0*Rz_0)
    nx_0 = Rx_0/len_R_0
    ny_0 = Ry_0/len_R_0
    nz_0 = Rz_0/len_R_0
    theta_0 = len_R_0
    sin_term_0 = np.sin(0.5*theta_0)

    Rx_0 = nx_0*sin_term_0
    Ry_0 = ny_0*sin_term_0
    Rz_0 = nz_0*sin_term_0
    Rq_0 = np.cos(0.5*theta_0)

    vs = solve_ivp(f_3, t_span = (t0, tf), y0 = [Rx_0, Ry_0, Rz_0, Rq_0], method='RK45', max_step=t_step_max, args=(w_version, w_params))

Rx_s = vs.y[0]
Ry_s = vs.y[1]
Rz_s = vs.y[2]
ts = vs.t
num_pts = len(ts)

wx_s, wy_s, wz_s = get_all_angular_velocities(ts, w_version, w_params)
w = np.array([wx_s, wy_s, wz_s])

if diffeq_method == 3:
    Rq_s = vs.y[3]
    theta_vec = 2*np.arccos(Rq_s)
    sin_term = np.sin(0.5*theta_vec)
    n_vec_x = Rx_s/sin_term
    n_vec_y = Ry_s/sin_term
    n_vec_z = Rz_s/sin_term
    Rx_s = n_vec_x*theta_vec
    Ry_s = n_vec_y*theta_vec
    Rz_s = n_vec_z*theta_vec
    # sanity check
    len_q = np.sqrt(Rx_s*Rx_s + Ry_s*Ry_s + Rz_s*Rz_s)
    nx = Rx_s / len_q
    ny = Ry_s / len_q
    nz = Rz_s / len_q
    Ex = nx*theta_vec
    Ey = ny*theta_vec
    Ez = nz*theta_vec

# Compute evolution of body vector as it rotates around w
# Bx = np.zeros(num_pts)
# By = np.zeros(num_pts)
# Bz = np.zeros(num_pts)
# Bx[0] = Bx_0
# By[0] = By_0
# Bz[0] = Bz_0
# for i in range(1, len(ts)):
#     dt = ts[i] - ts[i-1]
#     Bx[i] = Bx[i-1] + dt*(wy_s[i-1]*Bz[i-1] - By[i-1]*wz_s[i-1])
#     By[i] = By[i-1] + dt*(wz_s[i-1]*Bx[i-1] - Bz[i-1]*wx_s[i-1])
#     Bz[i] = Bz[i-1] + dt*(wx_s[i-1]*By[i-1] - Bx[i-1]*wy_s[i-1])
# B = np.array([Bx, By, Bz])

# B_length = np.zeros(num_pts)
# for i in range(num_pts):
#     B_length[i] = np.sqrt(Bx[i]*Bx[i] + By[i]*By[i] + Bz[i]*Bz[i])

# Compute evolution of body vector as it rotates around w
vs_b = solve_ivp(f_b, t_span = (t0, tf), y0 = [Bx_0, By_0, Bz_0], method='RK45', t_eval=ts, args=(w_version, w_params))
Bx = vs_b.y[0]
By = vs_b.y[1]
Bz = vs_b.y[2]
ts_b = vs_b.t
B = np.array([Bx, By, Bz])
B_length = np.zeros(num_pts)
for i in range(num_pts):
    B_length[i] = np.sqrt(Bx[i]*Bx[i] + By[i]*By[i] + Bz[i]*Bz[i])

len_R = np.sqrt(Rx_s*Rx_s + Ry_s*Ry_s + Rz_s*Rz_s)
nx = Rx_s/len_R
ny = Ry_s/len_R
nz = Rz_s/len_R
nx[np.isnan(nx)] = 0.0
ny[np.isnan(ny)] = 0.0
nz[np.isnan(nz)] = 0.0

if diffeq_method == 1 or diffeq_method == 3:
    phi_s = inverse_phi_fucntion(len_R)
elif diffeq_method == 2:
    phi_s = inverse_phi_fucntion_2(phi_glob, ts)

###  Compute divergence
w_dot_n = wx_s*nx + wy_s*ny + wz_s*nz
tan_func = 2*np.tan(0.5*phi_s)
div_f = w_dot_n*tan_func

# R_j = np.array([Rx_s[36100], Ry_s[36100], Rz_s[36100]])
# R_jj = (1.0/np.sqrt((R_j[0]**2 + R_j[1]**2 + R_j[2]**2)))*R_j

# w_j = np.array([wx_s[36100], wy_s[36100], wz_s[36100]])
# w_jj = (1.0/np.sqrt((w_j[0]**2 + w_j[1]**2 + w_j[2]**2)))*w_j

######################### Experiment - run backwards ############################################

run_backwards = False

if run_backwards:
    phi1 = phi_s[-1]
    n1x = nx[-1]
    n1y = ny[-1]
    n1z = nz[-1]

    phi_init = 2*np.pi - phi1
    n_init_x = -n1x
    n_init_y = -n1y
    n_init_z = -n1z

    R_init_x = phi_init*n_init_x
    R_init_y = phi_init*n_init_y
    R_init_z = phi_init*n_init_z

    w_version = 'experiment_a_2'

    # temp, try running longer
    tf = 10.0*tf

    vs_back = solve_ivp(f_1, t_span = (t0, tf), y0 = [R_init_x, R_init_y, R_init_z], method='RK45', max_step=t_step_max, args=(w_version, w_params))

    Rx_s2 = vs_back.y[0]
    Ry_s2 = vs_back.y[1]
    Rz_s2 = vs_back.y[2]
    ts2 = vs_back.t

    wx_s2, wy_s2, wz_s2 = get_all_angular_velocities(ts2, w_version, w_params)
    w2 = np.array([wx_s2, wy_s2, wz_s2])

    len_R2 = np.sqrt(Rx_s2*Rx_s2 + Ry_s2*Ry_s2 + Rz_s2*Rz_s2)
    nx2 = Rx_s2/len_R2
    ny2 = Ry_s2/len_R2
    nz2 = Rz_s2/len_R2
    # nx[np.isnan(nx2)] = 0.0
    # ny[np.isnan(ny2)] = 0.0
    # nz[np.isnan(nz2)] = 0.0

    phi_s2 = inverse_phi_fucntion(len_R2)

    plot_single_var_vs_time(ts2, phi_s2, tlim=(0, tf), xlim=(0, 20), xlabel='t', ylabel='|E|', add_h=True)

    phi_s2_max = np.max(phi_s2)

########################################################################################################

# # # Lyapunov exponent (time series method)
# lyap_estimate = lyapunov_exponent(Rx_0=Rx_0, Ry_0=Ry_0, Rz_0=Rz_0, t0=t0, tf=1000, t_step_max=t_step_max, t_start=0, delt=1e-3, w_version=w_version, w_params=w_params, use_equal_times=True)

# # Lyapunov exponent (Jacobian method)
# ly_exp_all = lyapunov_exponent_savary(Rx_0=Rx_0, Ry_0=Ry_0, Rz_0=Rz_0, t0=t0, t_step_max=t_step_max, averaging=True, n_steps=10**5)
# lyap_max = np.max(ly_exp_all)

# Zero-One test for chaos
# zero_one_test(Rx_s)

# # yy = 3*np.sin(ts) + np.sin(3*ts)
# # # plot_single_var_vs_time(ts, Rx_s, tlim=(0, tf), xlim=(-10, 5), xlabel='t', ylabel='Rx', title="Rx versus t")
# # # plot_single_var_vs_time(ts, phi_s, tlim=(0, tf), xlim=(0, 10), xlabel='t', ylabel='phi', title="phi versus t")
# # plot_single_var_vs_time(ts, yy, tlim=(0, tf), xlim=(-5, 5), xlabel='t', ylabel='test y', title="test y versus t")
# # # map_successive_max(Rx_s, var_name="Rx")
# # # map_successive_max(phi_s, var_name="phi")
# # map_successive_max(yy, var_name="test_y")

# # # Poincare section plot
# y_map, z_map, t_map = poincare_section(Rx_s, Ry_s, Rz_s, ts, normalized_lengths=False)

# # Strobe plot
# Rx_out, Ry_out, Rz_out, t_out = poincare_strobe(Rx_s, Ry_s, Rz_s, ts, T=(2.0*np.pi)/w1)

# # # # Strobe plot of many IC's
# # strobe_map_many(t0, tf, w_version, w_params, t_eval, t_step_max, use_equal_times, T=(2.0*np.pi)/w1, proj_2d=False)

# # Power spectrum for phi or z
# power_spectrum_auto(phi_s, t_step_max, xlim=[0.01, 0.5])

# # Recurrence plot
# recurrence_plot(Rx_s, Ry_s, Rz_s, ts, tf, max_time=2.0*100, norm_type='euclidean', eps=1.0, n_step=10, n_bins=30)

# Grid
x_grid = np.linspace(-7.0, 7.0, 10)  
y_grid = np.linspace(-7.0, 7.0, 10)  
z_grid = np.linspace(-7.0, 7.0, 10)  

# Set plot range
xlim=(-7.0, 7.0)
ylim=(-7.0, 7.0)
zlim=(-7.0, 7.0)

# # Plot a solution in a phase plane portrait in 2D
# plot_phase_sol(y, z, vs, (1,2), f_1, tfirst, xlim=ylim, ylim=zlim, add_phase_plane=False)

# # Plot time series of individual variables 
# plot_single_var_vs_time(ts, phi_s, tlim=(0, tf), xlim=(0, 8), xlabel='t', ylabel=r'$|E|$', title=r"$|E|$ versus t", add_h=True)
plot_single_var_vs_time(ts, phi_s, tlim=(0, tf), xlim=(0, 8), xlabel='t', ylabel=r'$|E|$', add_h=True)
plot_single_var_vs_time(ts, Rx_s, tlim=(0, tf), xlim=(-8, 8), xlabel='t', ylabel=r'$E_x$')
plot_single_var_vs_time(ts, Rz_s, tlim=(0, tf), xlim=(-8, 8), xlabel='t', ylabel=r'$E_z$')

# # phi max map
# map_successive_max(phi_s, var_name=r"$|E|$")

# Plot a solution in a phase plane portrait in 3D (not a true phase portrait if non-autonomous system)
if diffeq_method == 1:
    plot_phase_sol_3d(x_grid, y_grid, z_grid, vs, (0, 1, 2), f_1, tfirst, xlim=xlim, ylim=ylim, zlim=zlim, v_extra=[w, B], plot_normalized=False, add_phase_plane=False, title=None)
elif diffeq_method == 2:
    plot_phase_sol_3d(x_grid, y_grid, z_grid, vs, (0, 1, 2), f_2,  tfirst, xlim=xlim, ylim=ylim, zlim=zlim, v_extra=w, plot_normalized=False, add_phase_plane=False)
if diffeq_method == 3:
    plot_phase_sol_3d(x_grid, y_grid, z_grid, vs, (0, 1, 2), f_3, tfirst, xlim=xlim, ylim=ylim, zlim=zlim, v_extra=[w, B], plot_normalized=True, add_phase_plane=False, title=None)
plt.show()

# Plot animation
if diffeq_method == 1:
    plot_animation_soln_3d(x_grid, y_grid, z_grid, vs, (0, 1, 2), f_1, v_extra=[w, B], plot_normalized=True, tfirst=tfirst, xlim=xlim, ylim=ylim, zlim=zlim, interval=1, add_phase_plane=False, title='Animation of the Euler vector evolution', path="animation_euler_fast.gif")
elif diffeq_method == 2:
    plot_animation_soln_3d(x_grid, y_grid, z_grid, vs, (0, 1, 2), f_2, v_extra=w, plot_normalized=False, tfirst=tfirst, xlim=(-20,20), ylim=(-20,20), zlim=(-20,20), interval=200, add_phase_plane=False)

# # Plot w
# plot_angular_velocity(wx_s, wy_s, wz_s, dim=3)

# # Save figure in .eps and .png format
# plt.savefig('test.eps', format='eps')
# plt.savefig('test.png', format='png', dpi=300)

zzz = 1