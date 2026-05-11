# Davis Borders
# 05/11/2026
# 
# 

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------- Setup -------------------
# Physical constants
G = 6.674e-11                   # Gravitational constant (m^3 / kg * s^2)
m1 = 1e12                       # mass of body 1 (kg)
m2 = 1e12                       # mass of body 2 (kg)

# Initial Conditions
R1_0 = np.array([0, 0, 0])      # Initial position of mass 1 (m)
R1_DOT_0 = np.array([0, -0.41, 0])  # Initial velocity of mass 1 (m/s)
R2_0 = np.array([100, 0, 0])    # Initial position of mass 2 (m)
R2_DOT_0 = np.array([0, 0.41, 0]) # Initial velocity of mass 2 (m/s)


# Parameters
t_max = 1000                    # total time simulated (s)
dt = 0.01                       # step-size (s)

# Preallocation
num_steps = int(t_max / dt)
times = np.linspace(0, t_max, num_steps)
r1 = np.zeros((num_steps, 3))
r1_dot = np.zeros((num_steps, 3))
r1_ddot = np.zeros((num_steps, 3))
r2 = np.zeros((num_steps, 3))
r2_dot = np.zeros((num_steps, 3))
r2_ddot = np.zeros((num_steps, 3))

# ------------------- Physics -------------------
# Definition of ODE
def get_r1_ddot(r1, r2):
    dist = (np.linalg.norm(r2-r1))
    return G * m2/dist**3 * (r2-r1)
def get_r2_ddot(r1, r2):
    dist = (np.linalg.norm(r2-r1))
    return -G * m1/dist**3 * (r2-r1)

# Assign initial conditions
r1[0] = R1_0
r2[0] = R2_0
r1_dot[0] = R1_DOT_0
r2_dot[0] = R2_DOT_0

# Solution to Coupled-ODE
for i in range(num_steps - 1):
    r1_ddot[i] = get_r1_ddot(r1[i], r2[i])
    r2_ddot[i] = get_r2_ddot(r1[i], r2[i])

    # Euler-Cromer method
    r1_dot[i+1] = r1_dot[i] + r1_ddot[i]*dt
    r2_dot[i+1] = r2_dot[i] + r2_ddot[i]*dt
    r1[i+1] = r1[i] + r1_dot[i+1]*dt
    r2[i+1] = r2[i] + r2_dot[i+1]*dt

'''# ------------------- Visualization -------------------
plt.figure(figsize=(8, 8))
# plot both bodies
plt.plot(r1[:, 0], r1[:, 1], label="Mass 1", color="blue")
plt.plot(r2[:, 0], r2[:, 1], label="Mass 2", color="red")
# mark trajectories
plt.scatter(r1[0, 0], r1[0, 1], color="blue", marker="o")
plt.scatter(r2[0, 0], r2[0, 1], color="red", marker="o")
plt.title("2-Body Gravitational Orbit")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.grid(True)
plt.axis("equal") # equally weighs spatial dimensions
plt.show()'''

# Simulation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("2-Body Gravitational Orbit")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.grid(True)

# Set fixed axis limits based on the full trajectory
all_x = np.concatenate([r1[:, 0], r2[:, 0]])
all_y = np.concatenate([r1[:, 1], r2[:, 1]])
margin = 10
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
ax.set_aspect("equal")

# Trails (full path drawn so far)
trail1, = ax.plot([], [], color="blue", lw=1, label="Mass 1")
trail2, = ax.plot([], [], color="red",  lw=1, label="Mass 2")

# Dots (current position)
dot1, = ax.plot([], [], 'o', color="blue", ms=8)
dot2, = ax.plot([], [], 'o', color="red",  ms=8)

ax.legend()

def update(frame):
    trail1.set_data(r1[:frame, 0], r1[:frame, 1])
    trail2.set_data(r2[:frame, 0], r2[:frame, 1])
    dot1.set_data([r1[frame, 0]], [r1[frame, 1]])
    dot2.set_data([r2[frame, 0]], [r2[frame, 1]])
    return trail1, trail2, dot1, dot2

ani = FuncAnimation(
    fig,
    update,
    frames=range(0, num_steps, 100),  # 'step 10' skips frames to speed it up
    interval=20,                      # ms between frames
    blit=True
)

#ani.save("orbit.gif", fps=30)
plt.show()