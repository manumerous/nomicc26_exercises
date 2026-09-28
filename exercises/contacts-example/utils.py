import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nosnoc import latexify_plot
import numpy as np

def _circle(radius, center):
    tt = np.linspace(0, 2*np.pi, 100)
    return radius*np.cos(tt) + center[0], radius*np.sin(tt) + center[1]

def _wall(lims, q, orient="h"):
    tt = np.linspace(lims[0], lims[1], 100)
    if orient == "h":
        return tt,q*np.ones(100)
    elif orient == "v":
        return q*np.ones(100), tt
    else:
        raise NotImplementedError("Only h and v are supported orientations")

def _minkowski_quadratic_sphere(a,b,c,r,x):
    minimum = b/(2*a)

    parabola = np.vstack([x, a*x**2 + b*x + c])

    normal_parabola = np.vstack([-2*a*x - b,np.ones(x.shape)])
    normal_parabola /= np.linalg.norm(normal_parabola, axis=0)

    minkowski_diff = parabola + r*normal_parabola
    # TODO(@anton) this works for symmetric, that is b=0 but otherwise, it doesn't
    left_x = np.logical_and(x < minimum, minkowski_diff[0,:] > minimum)
    right_x = np.logical_and(x > minimum, minkowski_diff[0,:] < minimum)
    min_point = np.array([minimum, a*minimum**2 + b*minimum + c + r])
    minkowski_diff[:,left_x] = min_point[...,np.newaxis]
    minkowski_diff[:,right_x] = min_point[...,np.newaxis]
    return minkowski_diff[0,:],minkowski_diff[1,:]



def animate(x_res, q_target, opts, save_path=None, show=True):
    latexify_plot()
    q1 = x_res[:, 0:2]
    q2 = x_res[:, 2:4]
    q3 = x_res[:, 4:6]
    q_wall = x_res[:,6]

    q1_target = q_target[:, 0:2]
    q2_target = q_target[:, 2:4]
    q3_target = q_target[:, 4:6]

    n_frames = q1.shape[0]

    lim = np.max(np.abs(x_res[:, 0:6])) + 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.grid()

    # static elements
    # keepout for q3
    x = np.linspace(-2, 2, 100)
    ax.plot(*_minkowski_quadratic_sphere(10, 0, -4, opts.r3, x), "r:")

    # animated disc outlines and their traced paths
    disc1, = ax.plot([], [], "-", color="b", lw=2, label="disc 1")
    disc2, = ax.plot([], [], "-", color="b", lw=2, label="disc 2")
    disc3, = ax.plot([], [], "-", color="r", lw=2, label="disc 3")

    disc1_target, = ax.plot([], [], "-", color="b", lw=2, alpha=0.4, label="disc 1 target")
    disc2_target, = ax.plot([], [], "-", color="b", lw=2, alpha=0.4, label="disc 2 target")
    disc3_target, = ax.plot([], [], "-", color="r", lw=2, alpha=0.4, label="disc 3 target")

    wall, = ax.plot([], [], "-", color="k", lw=4, label="wall")
    trail1, = ax.plot([], [], "-", color="b", alpha=0.3)
    trail2, = ax.plot([], [], "-", color="b", alpha=0.3)
    trail3, = ax.plot([], [], "-", color="r", alpha=0.3)
    ax.legend(loc="upper right")

    def update(frame):
        disc1.set_data(*_circle(opts.r1, q1[frame]))
        disc2.set_data(*_circle(opts.r2, q2[frame]))
        disc3.set_data(*_circle(opts.r3, q3[frame]))

        disc1_target.set_data(*_circle(opts.r1, q1_target[int(frame/opts.N_fe)]))
        disc2_target.set_data(*_circle(opts.r2, q2_target[int(frame/opts.N_fe)]))
        disc3_target.set_data(*_circle(opts.r3, q3_target[int(frame/opts.N_fe)]))

        wall.set_data(*_wall([-lim,lim], q_wall[frame]))
        trail1.set_data(q1[:frame+1, 0], q1[:frame+1, 1])
        trail2.set_data(q2[:frame+1, 0], q2[:frame+1, 1])
        trail3.set_data(q3[:frame+1, 0], q3[:frame+1, 1])
        return disc1, disc2, disc3, disc1_target, disc2_target, disc3_target, wall, trail1, trail2, trail3

    h0 = opts.T/(opts.N*opts.N_fe)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000*h0, blit=True)

    if save_path is not None:
        anim.save(save_path, writer=PillowWriter(fps=max(1, int(1/h0))))
        print(f"  saved animation to {save_path}")
    if show:
        plt.show()
    return anim
