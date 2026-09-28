import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def plot_docking_trajectory(data, config, subtitle='', show=False):
    p = data[:, 0:3]
    v = data[:, 3:6]
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    speeds = np.linalg.norm(v, axis=1)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Trajectory colored by speed
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=speeds.min(), vmax=speeds.max())
    for i in range(len(x)-1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=cmap(norm(speeds[i])), alpha=0.8, lw=1.5)

    # --- DYNAMIC CONE ORIENTATION ---
    ef = config.docking_site_orientation / np.linalg.norm(config.docking_site_orientation)

    # Create local cone (pointing along Z)
    z_local = np.linspace(0, config.r_cone, 20)
    theta_local = np.linspace(0, 2 * np.pi, 20)
    theta_grid, z_grid = np.meshgrid(theta_local, z_local)
    r_at_z = z_grid * np.tan(config.theta_max)
    x_local = r_at_z * np.cos(theta_grid)
    y_local = r_at_z * np.sin(theta_grid)

    # Rotation Matrix to align [0,0,1] to ef
    # We use the Rodrigues rotation formula or a simple orthonormal basis
    z_axis = np.array([0, 0, 1])
    if np.allclose(ef, z_axis):
        rot_matrix = np.eye(3)
    elif np.allclose(ef, -z_axis):
        rot_matrix = np.diag([1, -1, -1])
    else:
        v = np.cross(z_axis, ef)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, ef)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))

    # Rotate and translate the cone vertices
    points = np.stack([x_local, y_local, z_grid], axis=-1)
    rotated_points = np.dot(points, rot_matrix.T)

    # Shift to target position
    rotated_points += config.docking_site_position

    # Plot the rotated cone surface
    ax.plot_surface(rotated_points[...,0], rotated_points[...,1], rotated_points[...,2],
                    color='cyan', alpha=0.2, linewidth=0)
    # --------------------------------

    ax.scatter(0, 0, 0, color='red', s=100, marker='X', label='Target', zorder=5)
    ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Start Position')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10).set_label('Speed [m/s]')

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title('Docking Trajectory with Oriented Cone')
    ax.legend()

    # Aspect ratio fix
    max_range = np.ptp(p, axis=0).max() / 2.0
    mid = np.mean(p, axis=0)
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

    plt.savefig(f"docking_trajectory_{subtitle}.png")
    if show:
        plt.show()

def plot_state_time_series(data, time, config, subtitle='', show=False):
    """
    Plots the 6 state variables and adds a vertical line for cone entry.
    """
    px, py, pz = data[:, 0], data[:, 1], data[:, 2]
    vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]

    states = [px, py, pz, vx, vy, vz]
    labels = ['Pos X [m]', 'Pos Y [m]', 'Pos Z [m]',
              'Vel X [m/s]', 'Vel Y [m/s]', 'Vel Z [m/s]']
    colors = ['blue', 'blue', 'blue', 'red', 'red', 'red']

    # --- NEW: CALCULATE CONE ENTRY TIME ---
    ef = config.docking_site_orientation
    rel_pos = data[:, 0:3] - config.docking_site_position
    dist = np.linalg.norm(rel_pos, axis=1)
    dot_prod = np.dot(rel_pos, ef)

    is_inside = (dist <= config.r_cone) & (dot_prod >= dist * np.cos(config.theta_max))
    entry_indices = np.where(is_inside)[0]
    entry_time = time[entry_indices[0]] if len(entry_indices) > 0 else None
    # ------------------------------------

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    axs_flat = axs.flatten()

    for i in range(6):
        ax = axs_flat[i]
        ax.plot(time, states[i], color=colors[i], lw=2)
        ax.set_ylabel(labels[i], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', lw=1, alpha=0.5)

        if i >= 3:
            ax.set_xlabel('Time [s]', fontsize=12)

        # Add the Cone Entry line to every plot
        if entry_time is not None:
            ax.axvline(entry_time, color='black', linestyle='--', lw=1.5, label='Cone Entry')

    plt.suptitle('Spacecraft State Evolution with Cone Entry', fontsize=16, y=1.02)

    # Add legend to the last plot to avoid repetition
    axs_flat[-1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"docking_state_timeseries_{subtitle}.png")
    if show:
        plt.show()


def plot_control_time_series(control_data, time, subtitle='', show=False):
    """
    Plots the control effort (forces) over time.
    Input:
        control_data: [N x 3] matrix (ux, uy, uz)
        time:         [N] array of timestamps
    """
    # Extract Control components
    ux, uy, uz = control_data[:, 0], control_data[:, 1], control_data[:, 2]

    # Setup for plotting
    controls = [ux, uy, uz]
    labels = ['Control Force X [N]', 'Control Force Y [N]', 'Control Force Z [N]']

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for i in range(3):
        ax = axs[i]
        ax.plot(time, controls[i], lw=2, marker='.')
        ax.set_ylabel(labels[i], fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.axhline(0, color='black', lw=1, alpha=0.5) # Zero reference line

        # Add a title to the first plot to distinguish it from the state plots
        if i == 0:
            ax.set_title('Control Effort over Time', fontsize=14)

    # X-axis label only for the bottom plot
    axs[2].set_xlabel('Time [s]', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"docking_control_timeseries_{subtitle}.png")
    if show:
        plt.show()


def plot_constraint_satisfaction(state, time, config, subtitle='', show=False):
    """
    Plots the validation of the two logical constraints that trigger
    when the spacecraft enters the distance radius r_cone.
    """
    # 1. Basic Extractions
    p = state[:, 0:3]
    v = state[:, 3:6]
    ef = config.docking_site_orientation / np.linalg.norm(config.docking_site_orientation)

    # 2. Calculate distances and norms
    rel_pos = p - config.docking_site_position
    dist = np.linalg.norm(rel_pos, axis=1)
    rel_pos[dist < 1e-5] = np.zeros(3)
    vel_norm = np.linalg.norm(v, axis=1)

    # --- CONSTRAINT 1: Line of Sight (LOS) ---
    # Constraint: (p-pf) dot ef / dist >= cos(theta)
    actual_cos = np.dot(rel_pos, ef) / dist
    los_residual = - actual_cos + np.cos(config.theta_max)

    # --- CONSTRAINT 2: Soft-Docking Velocity ---
    # Constraint: ||v|| - alpha * dist <= 0
    vel_residual = vel_norm - (config.alpha * dist)

    # --- TRIGGER: Find time when dist <= r_cone ---
    entry_indices = np.where(dist <= config.r_cone)[0]
    entry_time = time[entry_indices[0]] if len(entry_indices) > 0 else None

    # Create Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    # Plot 1: LOS Constraint
    ax1.plot(time, los_residual, color='blue', lw=2, label='LOS Residual')
    ax1.axhline(0, color='black', lw=2) # Threshold
    ax1.set_ylabel(r'$- \cos \theta$ + $\cos \theta_{max} <= ? 0 $',)
    ax1.set_title('Constraint 1: Line-of-Sight Orientation')
    ax1.grid(True, alpha=0.3)
    # Shade the "Satisfied" region (Above 0)
    ax1.fill_between(time, los_residual, 0, where=(los_residual <= 0), color='green', alpha=0.3)
    ax1.fill_between(time, los_residual, 0, where=(los_residual > 0), color='red', alpha=0.3)

    # Plot 2: Velocity Constraint
    ax2.plot(time, vel_residual, color='red', lw=2, label='Velocity Residual')
    ax2.axhline(0, color='black', lw=2) # Threshold
    ax2.set_ylabel(r'\n($\|v\| - \alpha \|p-p_f\| <= ? 0 $')
    ax2.set_title('Constraint 2: Soft-Docking Speed')
    ax2.set_xlabel('Time [s]')
    ax2.grid(True, alpha=0.3)
    # Shade the "Satisfied" region (Below 0)
    ax2.fill_between(time, vel_residual, 0, where=(vel_residual <= 0), color='green', alpha=0.3)
    ax2.fill_between(time, vel_residual, 0, where=(vel_residual > 0), color='red', alpha=0.3)

    # Add the vertical line for Distance Trigger
    if entry_time is not None:
        ax1.axvline(entry_time, color='black', linestyle='--', lw=2, label=f'Entry ($d < r_{{cone}}$)')
        ax2.axvline(entry_time, color='black', linestyle='--', lw=2)
        ax1.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"constraint_satisfaction_{subtitle}.png")
    if show:
        plt.show()
