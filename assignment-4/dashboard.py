import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import time
from matplotlib.patches import Circle
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.animation import FuncAnimation

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Robot Trajectory Planner",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- Load Model and Data ----------------
@st.cache_resource
def load_model_and_stats():
    """Load trained model and normalization statistics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    class TrajectoryMLP(nn.Module):
        def __init__(self, input_dim=7, output_dim=100, hidden_dims=[256, 512, 512, 256]):
            super(TrajectoryMLP, self).__init__()
            self.layers = nn.ModuleList()
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.layers.append(nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.3)
                ))
                prev_dim = hidden_dim
            self.output_layer = nn.Linear(prev_dim, output_dim)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.output_layer(x)

    try:
        checkpoint = torch.load('best_trajectory_model.pth', map_location=device)

        # Load training data for normalization stats
        with open('train_dataset.pkl', 'rb') as f:
            train_data = pickle.load(f)

        # Get normalization stats
        inputs = train_data['inputs']
        outputs = train_data['outputs']
        has_waypoint = (inputs[:, 4] >= 0).astype(np.float32)

        stats = {
            'input_mean': np.mean(inputs[:, :4], axis=0),
            'input_std': np.std(inputs[:, :4], axis=0) + 1e-8,
            'output_mean': np.mean(outputs, axis=0),
            'output_std': np.std(outputs, axis=0) + 1e-8,
            'metadata': train_data['metadata']
        }

        # Waypoint stats
        waypoint_data = inputs[has_waypoint == 1, 4:]
        if len(waypoint_data) > 0:
            stats['waypoint_mean'] = np.mean(waypoint_data, axis=0)
            stats['waypoint_std'] = np.std(waypoint_data, axis=0) + 1e-8
        else:
            stats['waypoint_mean'] = np.zeros(3)
            stats['waypoint_std'] = np.ones(3)

        model = TrajectoryMLP(input_dim=7, output_dim=100)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return model, stats, device, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False


# ---------------- Kinematics Functions ----------------
def forward_kinematics(q1, q2, L1=1.0, L2=1.0):
    """Compute end-effector position from joint angles"""
    x1 = L1 * np.cos(q1)
    y1 = L1 * np.sin(q1)
    x2 = x1 + L2 * np.cos(q1 + q2)
    y2 = y1 + L2 * np.sin(q1 + q2)
    return (x1, y1), (x2, y2)


def inverse_kinematics(x, y, L1=1.0, L2=1.0):
    """Compute joint angles from end-effector position (elbow-up solution)"""
    r = np.sqrt(x ** 2 + y ** 2)

    # Check if point is reachable
    if r > L1 + L2 or r < abs(L1 - L2):
        return None, None

    # Cosine law
    cos_q2 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q2 = np.clip(cos_q2, -1, 1)

    # Elbow-up solution
    q2 = np.arccos(cos_q2)

    # q1 using atan2
    alpha = np.arctan2(y, x)
    beta = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    q1 = alpha - beta

    return q1, q2


# ---------------- Optimization Functions ----------------
def quartic_trajectory(q_start, q_end, T, t):
    """Generate quartic trajectory"""
    a0 = q_start
    a1 = 0
    a2 = 0
    A = np.array([[T ** 3, T ** 4], [3 * T ** 2, 4 * T ** 3]])
    b = np.array([q_end - q_start, 0])
    a3, a4 = np.linalg.solve(A, b)
    return a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4


def optimize_trajectory(q1_start, q1_end, q2_start, q2_end,
                        wp_time_idx=None, wp_q1=None, wp_q2=None,
                        N=50, T=5.0):
    """Optimize trajectory with scipy (matches data generation)"""
    dt = T / (N - 1)
    t = np.linspace(0, T, N)

    # Initial guess
    q1_init = quartic_trajectory(q1_start, q1_end, T, t)
    q2_init = quartic_trajectory(q2_start, q2_end, T, t)
    x0 = np.hstack([q1_init, q2_init])

    def cost(x):
        q1 = x[:N]
        q2 = x[N:]

        # Acceleration cost
        acc_cost = 0.0
        for k in range(1, N - 1):
            q1_dd = (q1[k + 1] - 2 * q1[k] + q1[k - 1]) / dt ** 2
            q2_dd = (q2[k + 1] - 2 * q2[k] + q2[k - 1]) / dt ** 2
            acc_cost += q1_dd ** 2 + q2_dd ** 2

        # Velocity penalties
        v1_start = (q1[1] - q1[0]) / dt
        v1_end = (q1[-1] - q1[-2]) / dt
        v2_start = (q2[1] - q2[0]) / dt
        v2_end = (q2[-1] - q2[-2]) / dt
        vel_penalty = 1000.0 * (v1_start ** 2 + v1_end ** 2 + v2_start ** 2 + v2_end ** 2)

        # Position penalties
        pos_penalty = 10000.0 * ((q1[0] - q1_start) ** 2 + (q1[-1] - q1_end) ** 2 +
                                 (q2[0] - q2_start) ** 2 + (q2[-1] - q2_end) ** 2)

        # Waypoint penalty
        wp_cost = 0.0
        if wp_time_idx is not None:
            wp_cost = 5000.0 * ((q1[wp_time_idx] - wp_q1) ** 2 + (q2[wp_time_idx] - wp_q2) ** 2)

        return acc_cost + vel_penalty + pos_penalty + wp_cost

    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] - q1_start},
        {'type': 'eq', 'fun': lambda x: x[N - 1] - q1_end},
        {'type': 'eq', 'fun': lambda x: x[N] - q2_start},
        {'type': 'eq', 'fun': lambda x: x[2 * N - 1] - q2_end},
        {'type': 'eq', 'fun': lambda x: (x[1] - x[0]) / dt},
        {'type': 'eq', 'fun': lambda x: (x[N - 1] - x[N - 2]) / dt},
        {'type': 'eq', 'fun': lambda x: (x[N + 1] - x[N]) / dt},
        {'type': 'eq', 'fun': lambda x: (x[2 * N - 1] - x[2 * N - 2]) / dt}
    ]

    result = minimize(cost, x0, method='SLSQP', constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-9})

    return result.x[:N], result.x[N:]


def predict_trajectory(model, stats, device, q1_start, q1_end, q2_start, q2_end,
                       wp_time=None, wp_q1=None, wp_q2=None):
    """Predict trajectory using neural network"""
    if wp_time is not None and wp_time >= 0:
        input_vec = np.array([q1_start, q1_end, q2_start, q2_end, wp_time, wp_q1, wp_q2])
        input_vec[:4] = (input_vec[:4] - stats['input_mean']) / stats['input_std']
        input_vec[4:] = (input_vec[4:] - stats['waypoint_mean']) / stats['waypoint_std']
    else:
        input_vec = np.array([q1_start, q1_end, q2_start, q2_end, -1.0, 0.0, 0.0])
        input_vec[:4] = (input_vec[:4] - stats['input_mean']) / stats['input_std']

    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_vec).unsqueeze(0).to(device)
        output = model(input_tensor).cpu().numpy()[0]

    output = output * stats['output_std'] + stats['output_mean']
    N = len(output) // 2
    return output[:N], output[N:]


# ---------------- Animation Functions ----------------
def create_animation(q1_traj, q2_traj, L1=1.0, L2=1.0, title="Robot Animation", fps=60):
    """Create animation of robot arm motion"""
    N_original = len(q1_traj)
    T = 5.0
    N_target = int(fps * T)

    # Interpolate
    t_original = np.linspace(0, T, N_original)
    t_target = np.linspace(0, T, N_target)
    q1_interp = np.interp(t_target, t_original, q1_traj)
    q2_interp = np.interp(t_target, t_original, q2_traj)

    fig, ax = plt.subplots(figsize=(4.75, 4.75))

    ax.set_xlim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
    ax.set_ylim(-(L1 + L2) * 1.2, (L1 + L2) * 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    circle = Circle((0, 0), L1 + L2, fill=False, linestyle='--', color='lightgray', alpha=0.5)
    ax.add_patch(circle)

    link1, = ax.plot([], [], 'b-', linewidth=8, label='Link 1')
    link2, = ax.plot([], [], 'r-', linewidth=8, label='Link 2')
    base, = ax.plot([0], [0], 'ko', markersize=15, label='Base', zorder=5)
    elbow, = ax.plot([], [], 'go', markersize=12, label='Elbow', zorder=5)
    ee, = ax.plot([], [], 'ro', markersize=12, label='End-Effector', zorder=5)
    ee_path, = ax.plot([], [], 'r--', linewidth=1, alpha=0.3, label='Path')

    ee_trace_x = []
    ee_trace_y = []

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='upper right')

    def init():
        link1.set_data([], [])
        link2.set_data([], [])
        elbow.set_data([], [])
        ee.set_data([], [])
        ee_path.set_data([], [])
        time_text.set_text('')
        return link1, link2, elbow, ee, ee_path, time_text

    def animate(i):
        q1 = q1_interp[i]
        q2 = q2_interp[i]

        elbow_pos, ee_pos = forward_kinematics(q1, q2, L1, L2)

        link1.set_data([0, elbow_pos[0]], [0, elbow_pos[1]])
        link2.set_data([elbow_pos[0], ee_pos[0]], [elbow_pos[1], ee_pos[1]])
        elbow.set_data([elbow_pos[0]], [elbow_pos[1]])
        ee.set_data([ee_pos[0]], [ee_pos[1]])

        ee_trace_x.append(ee_pos[0])
        ee_trace_y.append(ee_pos[1])
        ee_path.set_data(ee_trace_x, ee_trace_y)

        current_time = i / fps
        time_text.set_text(f'Time: {current_time:.2f}s')

        return link1, link2, elbow, ee, ee_path, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=N_target,
                         interval=1000 / fps, blit=True, repeat=True)

    return fig, anim


# ---------------- Plotly Visualization Functions ----------------
def create_robot_arm_figure(q1, q2, L1=1.0, L2=1.0, title="Robot Arm"):
    """Create robot arm visualization"""
    elbow, ee = forward_kinematics(q1, q2, L1, L2)

    fig = go.Figure()

    theta = np.linspace(0, 2 * np.pi, 100)
    x_workspace = (L1 + L2) * np.cos(theta)
    y_workspace = (L1 + L2) * np.sin(theta)
    fig.add_trace(go.Scatter(x=x_workspace, y=y_workspace, mode='lines',
                             line=dict(color='lightgray', dash='dash'),
                             name='Workspace', hoverinfo='skip'))

    fig.add_trace(go.Scatter(x=[0, elbow[0]], y=[0, elbow[1]],
                             mode='lines+markers',
                             line=dict(color='blue', width=8),
                             marker=dict(size=12),
                             name='Link 1'))

    fig.add_trace(go.Scatter(x=[elbow[0], ee[0]], y=[elbow[1], ee[1]],
                             mode='lines+markers',
                             line=dict(color='red', width=8),
                             marker=dict(size=12),
                             name='Link 2'))

    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                             marker=dict(size=15, color='black'),
                             name='Base'))

    fig.update_xaxes(range=[-(L1 + L2) * 1.2, (L1 + L2) * 1.2], zeroline=True,
                     showgrid=True, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-(L1 + L2) * 1.2, (L1 + L2) * 1.2], zeroline=True, showgrid=True)

    fig.update_layout(height=400, title=title, showlegend=False, plot_bgcolor='white')

    return fig


# ---------------- Main Dashboard ----------------
def main():
    st.title("Robot Trajectory Planner")
    st.markdown("### Visual Interface for Trajectory Planning")

    model, stats, device, success = load_model_and_stats()

    if not success:
        st.error("Failed to load model.")
        st.stop()

    st.success("Model loaded successfully!")

    if 'start_x' not in st.session_state:
        st.session_state.start_x = 1.5
        st.session_state.start_y = 0.5
    if 'end_x' not in st.session_state:
        st.session_state.end_x = 0.5
        st.session_state.end_y = 1.5
    if 'waypoint_x' not in st.session_state:
        st.session_state.waypoint_x = 1.0
        st.session_state.waypoint_y = 1.0
    if 'use_waypoint' not in st.session_state:
        st.session_state.use_waypoint = False

    st.sidebar.header("Settings")
    L1 = st.sidebar.number_input("Link 1 Length (m)", 0.5, 2.0, 1.0, 0.1)
    L2 = st.sidebar.number_input("Link 2 Length (m)", 0.5, 2.0, 1.0, 0.1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Position Input")
    input_mode = st.sidebar.radio("Input Method:",
                                  ["Manual (X, Y)", "Joint Angles"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Start Position")
        if input_mode == "Manual (X, Y)":
            start_x = st.number_input("Start X (m)", -(L1 + L2), L1 + L2,
                                      st.session_state.start_x, 0.1)
            start_y = st.number_input("Start Y (m)", -(L1 + L2), L1 + L2,
                                      st.session_state.start_y, 0.1)
            q1_start, q2_start = inverse_kinematics(start_x, start_y, L1, L2)
            if q1_start is None:
                st.error("Position unreachable!")
                q1_start, q2_start = 0.0, 0.0
            else:
                st.session_state.start_x = start_x
                st.session_state.start_y = start_y
                st.info(f"q1 = {q1_start:.3f} rad, q2 = {q2_start:.3f} rad")
        else:
            q1_start = st.slider("Joint 1 (rad)", -np.pi, np.pi, 0.0, 0.1)
            q2_start = st.slider("Joint 2 (rad)", -np.pi / 2, np.pi / 2, 0.0, 0.1)

        fig_start = create_robot_arm_figure(q1_start, q2_start, L1, L2, "Start Configuration")
        st.plotly_chart(fig_start, use_container_width=True)

    with col2:
        st.subheader("End Position")
        if input_mode == "Manual (X, Y)":
            end_x = st.number_input("End X (m)", -(L1 + L2), L1 + L2,
                                    st.session_state.end_x, 0.1)
            end_y = st.number_input("End Y (m)", -(L1 + L2), L1 + L2,
                                    st.session_state.end_y, 0.1)
            q1_end, q2_end = inverse_kinematics(end_x, end_y, L1, L2)
            if q1_end is None:
                st.error("Position unreachable!")
                q1_end, q2_end = 0.0, 0.0
            else:
                st.session_state.end_x = end_x
                st.session_state.end_y = end_y
                st.info(f"q1 = {q1_end:.3f} rad, q2 = {q2_end:.3f} rad")
        else:
            q1_end = st.slider("Joint 1 (rad) ", -np.pi, np.pi, np.pi / 2, 0.1)
            q2_end = st.slider("Joint 2 (rad) ", -np.pi / 2, np.pi / 2, np.pi / 3, 0.1)

        fig_end = create_robot_arm_figure(q1_end, q2_end, L1, L2, "End Configuration")
        st.plotly_chart(fig_end, use_container_width=True)

    st.markdown("---")
    use_waypoint = st.checkbox("Add Waypoint", value=st.session_state.use_waypoint)
    st.session_state.use_waypoint = use_waypoint

    wp_time = None
    wp_q1 = None
    wp_q2 = None
    wp_time_idx = None

    if use_waypoint:
        col1, col2 = st.columns([2, 1])

        with col1:
            if input_mode == "Manual (X, Y)":
                wp_x = st.number_input("Waypoint X (m)", -(L1 + L2), L1 + L2,
                                       st.session_state.waypoint_x, 0.1)
                wp_y = st.number_input("Waypoint Y (m)", -(L1 + L2), L1 + L2,
                                       st.session_state.waypoint_y, 0.1)
                wp_q1, wp_q2 = inverse_kinematics(wp_x, wp_y, L1, L2)
                if wp_q1 is None:
                    st.error("Waypoint unreachable!")
                    wp_q1, wp_q2 = 0.0, 0.0
                else:
                    st.session_state.waypoint_x = wp_x
                    st.session_state.waypoint_y = wp_y
            else:
                wp_q1 = st.slider("Waypoint Joint 1 (rad)", -np.pi, np.pi, 0.5, 0.1)
                wp_q2 = st.slider("Waypoint Joint 2 (rad)", -np.pi / 2, np.pi / 2, 0.3, 0.1)

        with col2:
            T = stats['metadata']['T']
            N = stats['metadata']['N']
            wp_time = st.slider("Waypoint Time (s)", 0.5, T - 0.5, T / 2, 0.1)
            wp_time_idx = int((wp_time / T) * (N - 1))

            fig_wp = create_robot_arm_figure(wp_q1, wp_q2, L1, L2, "Waypoint")
            st.plotly_chart(fig_wp, use_container_width=True)

    st.markdown("---")
    if st.button("Generate Trajectories", type="primary", use_container_width=True):
        N = stats['metadata']['N']
        T = stats['metadata']['T']
        t = np.linspace(0, T, N)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Optimization")
            with st.spinner("Running..."):
                start_time = time.time()
                q1_opt, q2_opt = optimize_trajectory(
                    q1_start, q1_end, q2_start, q2_end,
                    wp_time_idx, wp_q1, wp_q2, N, T
                )
                opt_time = time.time() - start_time

        with col2:
            st.subheader("Neural Network")
            with st.spinner("Running..."):
                start_time = time.time()
                q1_pred, q2_pred = predict_trajectory(
                    model, stats, device,
                    q1_start, q1_end, q2_start, q2_end,
                    wp_time, wp_q1, wp_q2
                )
                nn_time = time.time() - start_time

        rmse = np.sqrt(np.mean((np.hstack([q1_opt, q2_opt]) -
                                np.hstack([q1_pred, q2_pred])) ** 2))

        st.markdown("---")
        st.metric("Prediction Error (RMSE)",
                  f"{rmse:.4f} rad", f"{np.degrees(rmse):.2f} degrees")

        st.markdown("---")
        st.header("Robot Motion Animation")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Optimized Trajectory")
            with st.spinner("Generating animation..."):
                fig_anim_opt, anim_opt = create_animation(q1_opt, q2_opt, L1, L2,
                                                          "Optimized Trajectory", fps=30)
                html_str = anim_opt.to_jshtml()
                st.components.v1.html(html_str, height=550, scrolling=False)

        with col2:
            st.subheader("Neural Network Prediction")
            with st.spinner("Generating animation..."):
                fig_anim_pred, anim_pred = create_animation(q1_pred, q2_pred, L1, L2,
                                                            "NN Predicted Trajectory", fps=30)
                html_str = anim_pred.to_jshtml()
                st.components.v1.html(html_str, height=550, scrolling=False)

        plt.close('all')

        st.markdown("---")
        st.header("Joint Trajectories")

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=("Joint 1", "Joint 2"))

        fig.add_trace(go.Scatter(x=t, y=q1_opt, name='Optimized',
                                 line=dict(color='blue', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=q1_pred, name='NN Predicted',
                                 line=dict(color='red', width=2, dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=t, y=q2_opt, name='Optimized',
                                 line=dict(color='blue', width=3), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=q2_pred, name='NN Predicted',
                                 line=dict(color='red', width=2, dash='dash'), showlegend=False), row=2, col=1)

        if use_waypoint:
            fig.add_trace(go.Scatter(x=[wp_time], y=[wp_q1], mode='markers',
                                     marker=dict(size=15, symbol='star', color='gold'),
                                     name='Waypoint'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[wp_time], y=[wp_q2], mode='markers',
                                     marker=dict(size=15, symbol='star', color='gold'),
                                     showlegend=False), row=2, col=1)

        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="q1 (rad)", row=1, col=1)
        fig.update_yaxes(title_text="q2 (rad)", row=2, col=1)
        fig.update_layout(height=600, title_text="Joint Trajectories")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.header("End-Effector Path")

        ee_opt = np.array([forward_kinematics(q1, q2, L1, L2)[1]
                           for q1, q2 in zip(q1_opt, q2_opt)])
        ee_pred = np.array([forward_kinematics(q1, q2, L1, L2)[1]
                            for q1, q2 in zip(q1_pred, q2_pred)])

        fig_ee = go.Figure()

        theta = np.linspace(0, 2 * np.pi, 100)
        fig_ee.add_trace(go.Scatter(
            x=(L1 + L2) * np.cos(theta), y=(L1 + L2) * np.sin(theta),
            mode='lines', line=dict(color='lightgray', dash='dash'),
            name='Workspace', hoverinfo='skip'))

        fig_ee.add_trace(go.Scatter(x=ee_opt[:, 0], y=ee_opt[:, 1],
                                    mode='lines', line=dict(color='blue', width=3),
                                    name='Optimized'))
        fig_ee.add_trace(go.Scatter(x=ee_pred[:, 0], y=ee_pred[:, 1],
                                    mode='lines', line=dict(color='red', width=2, dash='dash'),
                                    name='NN Predicted'))

        fig_ee.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                                    marker=dict(size=15, color='black'),
                                    name='Base'))
        fig_ee.add_trace(go.Scatter(x=[ee_opt[0, 0]], y=[ee_opt[0, 1]],
                                    mode='markers', marker=dict(size=12, color='green'),
                                    name='Start'))
        fig_ee.add_trace(go.Scatter(x=[ee_opt[-1, 0]], y=[ee_opt[-1, 1]],
                                    mode='markers', marker=dict(size=12, color='red'),
                                    name='End'))

        fig_ee.update_xaxes(range=[-(L1 + L2) * 1.2, (L1 + L2) * 1.2], zeroline=True,
                            showgrid=True, scaleanchor="y", scaleratio=1)
        fig_ee.update_yaxes(range=[-(L1 + L2) * 1.2, (L1 + L2) * 1.2], zeroline=True, showgrid=True)
        fig_ee.update_layout(height=600, title="End-Effector Path", plot_bgcolor='white')

        st.plotly_chart(fig_ee, use_container_width=True)


if __name__ == "__main__":
    main()