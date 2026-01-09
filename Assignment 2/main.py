import math
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
T = 2.0          # Total time (seconds)
N = 400          # Number of time samples
dt = T / (N - 1)

t = [i * dt for i in range(N)]

# Initial and final joint angles (radians)
q1_start, q1_end = 0.0, math.pi / 2
q2_start, q2_end = 0.0, math.pi / 3

# ---------------- Linear Trajectory ----------------
def linear_trajectory(q_start, q_end):
    return [q_start + (q_end - q_start) * (ti / T) for ti in t]

q1_linear = linear_trajectory(q1_start, q1_end)
q2_linear = linear_trajectory(q2_start, q2_end)

# ---------------- Cubic Polynomial Trajectory ----------------
def cubic_trajectory(q_start, q_end):
    a0 = q_start
    a1 = 0
    a2 = 3 * (q_end - q_start) / (T ** 2)
    a3 = -2 * (q_end - q_start) / (T ** 3)

    q = [a0 + a1*ti + a2*ti**2 + a3*ti**3 for ti in t]
    q_dot = [a1 + 2*a2*ti + 3*a3*ti**2 for ti in t]

    return q, q_dot

q1_poly, q1_dot_poly = cubic_trajectory(q1_start, q1_end)
q2_poly, q2_dot_poly = cubic_trajectory(q2_start, q2_end)

# ---------------- Linear Velocity (Numerical) ----------------
def numerical_velocity(q):
    return [(q[i+1] - q[i]) / dt for i in range(len(q)-1)] + [0]

q1_dot_linear = numerical_velocity(q1_linear)
q2_dot_linear = numerical_velocity(q2_linear)

# ---------------- One Big Figure ----------------
plt.figure(figsize=(12, 8))

# q1 angle
plt.subplot(2, 2, 1)
plt.plot(t, q1_linear, 'r--', label='q1 Linear')
plt.plot(t, q1_poly, 'r', label='q1 Polynomial')
plt.ylabel("q1 (rad)")
plt.title("Joint 1 Angle")
plt.legend()
plt.grid(True)

# q2 angle
plt.subplot(2, 2, 2)
plt.plot(t, q2_linear, 'b--', label='q2 Linear')
plt.plot(t, q2_poly, 'b', label='q2 Polynomial')
plt.ylabel("q2 (rad)")
plt.title("Joint 2 Angle")
plt.legend()
plt.grid(True)

# q1 velocity
plt.subplot(2, 2, 3)
plt.plot(t, q1_dot_linear, 'r--', label='q1 Linear Velocity')
plt.plot(t, q1_dot_poly, 'r', label='q1 Polynomial Velocity')
plt.xlabel("Time (s)")
plt.ylabel("q1 dot (rad/s)")
plt.title("Joint 1 Velocity")
plt.legend()
plt.grid(True)

# q2 velocity
plt.subplot(2, 2, 4)
plt.plot(t, q2_dot_linear, 'b--', label='q2 Linear Velocity')
plt.plot(t, q2_dot_poly, 'b', label='q2 Polynomial Velocity')
plt.xlabel("Time (s)")
plt.ylabel("q2 dot (rad/s)")
plt.title("Joint 2 Velocity")
plt.legend()
plt.grid(True)

plt.suptitle("Joint-Space Trajectory Comparison: Linear vs Polynomial", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
