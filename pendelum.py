# Davis Borders
# 05/05/2026
# This program uses numerical methods to solve a simple ODE: the damped-nonlinear pendelum. The program highlights numerical methods (specifically the Euler-Cromer method), phase-space, and implicitly hints at chaos theory.
# This program is meant to serve as a guide to more complex differntial equations, as the methodolgy is the same. Furthermore, additional logic can be implemented like animations using the solutions computed.

# Imports
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Setup -------------------
# Physical constants
g = 9.81                # gravitational constant (m/s^2)
L = 1                   # length of pendelum string (m)
mu = 0.1                # coefficent of friction (1/s)

# Initial conditions
THETA_0 = np.pi/3       # initial angle of pendelulm relative to the negative y-axis (rad)
THETA_DOT_0 = 0         # initial angular speed (rad/s)

# Parameters
t_max = 100             # total time simulated (s)
delta_t = 0.01          # step-size (s)

# Preallocation
num_steps = int(t_max / delta_t)    # It's good practice to use num_steps over delta_t due to truncation 
times = np.linspace(0, t_max, num_steps)
theta = np.zeros(num_steps)
theta_dot = np.zeros(num_steps)
theta_dot_dot = np.zeros(num_steps)

# ------------------- Physics -------------------
# Definition of ODE
def get_theta_double_dot(theta, theta_dot):
        return -mu * theta_dot - (g/L)*np.sin(theta)

# Assign initial conditions
theta[0] = THETA_0
theta_dot[0] = THETA_DOT_0

# Solution to ODE
for i in range(num_steps - 1):
    theta_dot_dot[i] = get_theta_double_dot(theta[i], theta_dot[i])
    # Euler-Cromer method (better converses energy)     --->    EC method follows form of next_state = current_state + rate_of_change_of_current * dt
    theta_dot[i+1] =  theta_dot[i] + theta_dot_dot[i] * delta_t
    theta[i+1] = theta[i] + theta_dot[i+1] * delta_t

# ------------------- Visualization -------------------
# Angle vs. Time Plot
plt.plot(times, theta)
plt.title("Damped Pendelum Simulation")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.grid(True)
plt.show()

# Phase-space Plot
plt.plot(theta, theta_dot)
plt.title("Phase-Space")
plt.xlabel("Angle (rad)")
plt.ylabel("Angular Velocity (rad/s)")
plt.grid(True)
plt.show()