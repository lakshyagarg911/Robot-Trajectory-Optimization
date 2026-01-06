import math
import matplotlib.pyplot as plt

# Link lengths
l1 = 1
l2 = 1

def xy_calculation(q1, q2):
    # Elbow position
    x1 = l1 * math.cos(q1)
    y1 = l1 * math.sin(q1)

    # End-effector position
    x2 = x1 + l2 * math.cos(q1 + q2)
    y2 = y1 + l2 * math.sin(q1 + q2)

    return (x1, y1), (x2, y2)

def plot_arm(q1, q2, title):
    elbow, ee = xy_calculation(q1, q2)

    # Base, elbow, end-effector
    x0, y0 = 0, 0
    x1, y1 = elbow
    x2, y2 = ee

    # Link 1: Base -> Elbow (Blue)
    plt.plot([x0, x1], [y0, y1],
             color='blue', linewidth=4, label='Link 1')

    # Link 2: Elbow -> End Effector (Red)
    plt.plot([x1, x2], [y1, y2],
             color='red', linewidth=4, label='Link 2')

    # Joints
    plt.scatter([x0, x1, x2], [y0, y1, y2],
                color='black', s=80, zorder=5)

    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.axis('equal')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

configs = [
    (0, 0, "Straight"),
    (math.pi/4, 3 *     math.pi/8, "bent elbow"),
    (math.pi/48, 46 * math.pi/48, "folded")
]

plt.figure(figsize=(12, 4))

for i, (q1, q2, title) in enumerate(configs, 1):
    plt.subplot(1, 3, i)
    plot_arm(q1, q2, title)

plt.tight_layout()
plt.show()
