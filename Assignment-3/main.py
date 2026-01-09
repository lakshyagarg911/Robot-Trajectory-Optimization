import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------------- Parameters ----------------
T = 5.0
N = 50
dt = T / (N - 1)
t = np.linspace(0, T, N)

# Start and end joint angles
q1_start, q1_end = 0.0, np.pi / 2
q2_start, q2_end = 0.0, np.pi / 3

# ---------------- Quartic Trajectory ----------------
def quartic_trajectory(q_start, q_end):
    a0 = q_start
    a1 = 0
    a2 = 0

    A = np.array([
        [T**3, T**4],
        [3*T**2, 4*T**3]
    ])
    b = np.array([
        q_end - q_start,
        0
    ])

    a3, a4 = np.linalg.solve(A, b)

    q = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4
    q_dot = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3
    q_ddot = 2*a2 + 6*a3*t + 12*a4*t**2

    return q, q_dot, q_ddot

# Quartic baseline
q1_q, q1_dot, q1_ddot = quartic_trajectory(q1_start, q1_end)
q2_q, q2_dot, q2_ddot = quartic_trajectory(q2_start, q2_end)

# ---------------- Acceleration Cost Optimization ----------------
def acceleration_cost(x):
    q1 = x[:N]
    q2 = x[N:]

    cost = 0.0
    for k in range(1, N - 1):
        q1_dd = (q1[k+1] - 2*q1[k] + q1[k-1]) / dt**2
        q2_dd = (q2[k+1] - 2*q2[k] + q2[k-1]) / dt**2
        cost += q1_dd**2 + q2_dd**2
    return cost

# Initial guess
x0 = np.hstack([q1_q, q2_q])

constraints = [
    {'type': 'eq', 'fun': lambda x: x[0] - q1_start},
    {'type': 'eq', 'fun': lambda x: x[N-1] - q1_end},
    {'type': 'eq', 'fun': lambda x: x[N] - q2_start},
    {'type': 'eq', 'fun': lambda x: x[2*N-1] - q2_end}
]

result = minimize(acceleration_cost, x0, method='SLSQP', constraints=constraints)

q1_opt = result.x[:N]
q2_opt = result.x[N:]

# ---------------- Velocities and Accelerations ----------------
def velocity(q):
    return np.gradient(q, dt)

def acceleration(q):
    return np.gradient(np.gradient(q, dt), dt)

q1_dot_opt = velocity(q1_opt)
q2_dot_opt = velocity(q2_opt)

q1_ddot_opt = acceleration(q1_opt)
q2_ddot_opt = acceleration(q2_opt)

# ---------------- Combined Plot ----------------
plt.figure(figsize=(14, 10))

# Angles
plt.subplot(3, 2, 1)
plt.plot(t, q1_q, 'r--', label='Quartic')
plt.plot(t, q1_opt, 'r', label='Optimized')
plt.ylabel("q1 (rad)")
plt.title("Joint 1 Angle")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(t, q2_q, 'b--', label='Quartic')
plt.plot(t, q2_opt, 'b', label='Optimized')
plt.ylabel("q2 (rad)")
plt.title("Joint 2 Angle")
plt.legend()
plt.grid(True)

# Velocities
plt.subplot(3, 2, 3)
plt.plot(t, q1_dot, 'r--', label='Quartic')
plt.plot(t, q1_dot_opt, 'r', label='Optimized')
plt.ylabel("q1 dot (rad/s)")
plt.title("Joint 1 Velocity")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(t, q2_dot, 'b--', label='Quartic')
plt.plot(t, q2_dot_opt, 'b', label='Optimized')
plt.ylabel("q2 dot (rad/s)")
plt.title("Joint 2 Velocity")
plt.legend()
plt.grid(True)

# Accelerations
plt.subplot(3, 2, 5)
plt.plot(t, q1_ddot, 'r--', label='Quartic')
plt.plot(t, q1_ddot_opt, 'r', label='Optimized')
plt.xlabel("Time (s)")
plt.ylabel("q1 ddot (rad/s²)")
plt.title("Joint 1 Acceleration")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(t, q2_ddot, 'b--', label='Quartic')
plt.plot(t, q2_ddot_opt, 'b', label='Optimized')
plt.xlabel("Time (s)")
plt.ylabel("q2 ddot (rad/s²)")
plt.title("Joint 2 Acceleration")
plt.legend()
plt.grid(True)

plt.suptitle("Quartic vs Acceleration-Optimized Joint Trajectories", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
