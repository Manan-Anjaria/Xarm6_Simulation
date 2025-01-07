import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import math as m
import time

# Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # To load plane and other objects
p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0,0,0])

# Load the plane and robot
plane_id = p.loadURDF("plane.urdf")
# Replace 'your_robot.urdf' with the path to your robot's URDF file
robot_urdf_path = r"C:\Users\anjar\OneDrive\Desktop\FOR\bullet3-master\examples\pybullet\gym\pybullet_data\xarm\xarm6_robot.urdf"
robot_id = p.loadURDF(robot_urdf_path, useFixedBase=True)

# Define initial and final joint angles in degrees
initial = np.array([-50, -10, -110, -50, -50, 50])
final = np.array([-10, -10, -10, -50, -50, 50])

# Convert joint angles to radians for simulation
initial_joint_angles = [m.radians(angle) for angle in initial]
final_joint_angles = [m.radians(angle) for angle in final]

# Display initial and final joint angles
print("Initial Joint Angles (degrees):", initial)
print("Final Joint Angles (degrees):", final)
print("\n")

# Define trajectory duration and time parameters
tf = 5    # Final time in seconds
t0 = 0    # Initial time
num_points = 100

# Generate time vector
t = np.linspace(t0, tf, num_points)

# Initialize joint trajectory arrays for different methods
joint_trajectory_linear = np.zeros((num_points, len(initial)))
joint_trajectory_cubic = np.zeros((num_points, len(initial)))
joint_trajectory_quintic = np.zeros((num_points, len(initial)))
joint_trajectory_bspline = np.zeros((num_points, len(initial)))
joint_trajectory_bezier = np.zeros((num_points, len(initial)))
joint_trajectory_trapezoidal = np.zeros((num_points, len(initial)))

# Initialize lists to store maximum accelerations for each joint (optional)
max_acc_cubic = []
max_acc_quintic = []
max_acc_trapezoidal = []

# -------------------- Linear Interpolation --------------------
print("----- Linear Interpolation Calculations -----")
for i in range(len(initial)):
    q0 = initial[i]
    qf = final[i]
    slope = (qf - q0) / tf
    intercept = q0
    print(f"Joint {i+1}: q(t) = {intercept:.2f} + {slope:.2f} * t")
    
    # Linear interpolation: q(t) = q0 + (qf - q0) * (t / tf)
    joint_trajectory_linear[:, i] = q0 + (qf - q0) * (t / tf)
print("\n")

# -------------------- Cubic Polynomial --------------------
print("----- Cubic Polynomial Calculations -----")
for i in range(len(initial)):
    q0 = initial[i]
    qf = final[i]
    a0 = q0
    a1 = 0
    a2 = 3 * (qf - q0) / (tf**2)
    a3 = -2 * (qf - q0) / (tf**3)
    
    # Calculate and store maximum acceleration for each joint
    q_ddot_tf = 2*a2 + 6*a3*tf
    max_acc = abs(q_ddot_tf)
    max_acc_cubic.append(max_acc)
    
    print(f"Joint {i+1}:")
    print(f"  q(t) = {a0:.2f} + {a1:.2f} * t + {a2:.2f} * t^2 + {a3:.2f} * t^3")
    print(f"  Maximum Acceleration at t={tf}s: {max_acc:.2f} deg/s²")
    
    # Evaluate cubic polynomial for each time point: q(t) = a0 + a1*t + a2*t^2 + a3*t^3
    joint_trajectory_cubic[:, i] = a0 + a1 * t + a2 * t**2 + a3 * t**3
print("\n")

# -------------------- Quintic Polynomial --------------------
print("----- Quintic Polynomial Calculations -----")
def compute_quintic_coefficients(q0, qf, tf):
    # Boundary conditions: q0, qf, v0=0, vf=0, a0=0, af=0
    A = np.array([
        [1, 0,      0,       0,        0,         0],
        [0, 1,      0,       0,        0,         0],
        [0, 0,      2,       0,        0,         0],
        [1, tf, tf**2, tf**3, tf**4, tf**5],
        [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
        [0, 0,      2,    6*tf,   12*tf**2, 20*tf**3]
    ])
    B = np.array([q0, 0, 0, qf, 0, 0])
    coeffs = np.linalg.solve(A, B)
    return coeffs

for i in range(len(initial)):
    q0 = initial[i]
    qf = final[i]
    coeffs = compute_quintic_coefficients(q0, qf, tf)
    
    # Calculate acceleration coefficients
    a0_coeff = coeffs[0]
    a1_coeff = coeffs[1]
    a2_coeff = coeffs[2]
    a3_coeff = coeffs[3]
    a4_coeff = coeffs[4]
    a5_coeff = coeffs[5]
    q_double_dot_tf = 2*a2_coeff + 6*a3_coeff*tf + 12*a4_coeff*tf**2 + 20*a5_coeff*tf**3
    max_acc_quintic.append(abs(q_double_dot_tf))
    
    print(f"Joint {i+1}:")
    print(f"  q(t) = {a0_coeff:.2f} + {a1_coeff:.2f} * t + {a2_coeff:.2f} * t^2 + {a3_coeff:.2f} * t^3 + {a4_coeff:.2f} * t^4 + {a5_coeff:.2f} * t^5")
    print(f"  Maximum Acceleration at t={tf}s: {abs(q_double_dot_tf):.2f} deg/s²")
    
    # Evaluate quintic polynomial for each time point
    joint_trajectory_quintic[:, i] = (
        coeffs[0] +
        coeffs[1] * t +
        coeffs[2] * t**2 +
        coeffs[3] * t**3 +
        coeffs[4] * t**4 +
        coeffs[5] * t**5
    )
print("\n")

# -------------------- B-Spline (Quadratic) --------------------
print("----- B-Spline (Quadratic) Calculations -----")
for i in range(len(initial)):
    # Define control points: start, mid, end
    q0 = initial[i]
    qf = final[i]
    qm = (q0 + qf) / 2  # Midpoint
    
    # Knot vector for quadratic B-Spline with clamped ends
    knots = [0, 0, 0, 1, 1, 1]
    
    # B-Spline basis functions for k=2
    def basis(k, j, t_val, knots):
        if k == 0:
            return 1.0 if knots[j] <= t_val < knots[j+1] else 0.0
        else:
            denom1 = knots[j+k] - knots[j]
            denom2 = knots[j+k+1] - knots[j+1]
            term1 = 0.0
            term2 = 0.0
            if denom1 != 0:
                term1 = ((t_val - knots[j]) / denom1) * basis(k-1, j, t_val, knots)
            if denom2 != 0:
                term2 = ((knots[j+k+1] - t_val) / denom2) * basis(k-1, j+1, t_val, knots)
            return term1 + term2
    
    # Evaluate B-Spline at each time point
    spline_values = np.zeros(num_points)
    for idx, ti in enumerate(t):
        # Normalize ti to [0,1] for knot vector
        normalized_t = (ti - t0) / (tf - t0)
        spline = 0.0
        for j in range(3):
            spline += [q0, qm, qf][j] * basis(2, j, normalized_t, knots)
        spline_values[idx] = spline
    joint_trajectory_bspline[:, i] = spline_values
    
    print(f"Joint {i+1}:")
    print(f"  Control Points: P0 = {q0}, P1 = {qm}, P2 = {qf}")
    print(f"  B-Spline Equation: q(t) = P0 * B0,2(t) + P1 * B1,2(t) + P2 * B2,2(t)")
print("\n")

# -------------------- Bezier Curve (Linear) --------------------
print("----- Bezier Curve (Linear) Calculations -----")
for i in range(len(initial)):
    q0 = initial[i]
    qf = final[i]
    
    # Linear Bezier: B(t) = (1 - t/tf)*q0 + (t/tf)*qf
    bezier_values = (1 - t/tf) * q0 + (t/tf) * qf
    joint_trajectory_bezier[:, i] = bezier_values
    
    print(f"Joint {i+1}:")
    print(f"  Control Points: P0 = {q0}, P1 = {qf}")
    print(f"  Bezier Curve Equation: B(t) = (1 - t/tf)*{q0} + (t/tf)*{qf}")
print("\n")

# -------------------- Trapezoidal Velocity Profile --------------------
print("----- Trapezoidal Velocity Profile Calculations -----")
# Define maximum velocity and acceleration for trapezoidal profile
V_max = np.array([10, 10, 10, 10, 10, 10])  # degrees per second
A_max = np.array([20, 20, 20, 20, 20, 20])  # degrees per second squared

def trapezoidal_profile(q0, qf, V_max, A_max, tf, t):
    D = qf - q0
    direction = 1 if D >= 0 else -1
    D = abs(D)
    V_max = abs(V_max)
    A_max = abs(A_max)
    
    # Time to accelerate to V_max
    t_accel = V_max / A_max
    # Distance covered during acceleration
    D_accel = 0.5 * A_max * t_accel**2
    
    # Check if the profile is trapezoidal or triangular
    if 2 * D_accel > D:
        # Triangle profile (no cruise phase)
        t_accel = np.sqrt(D / A_max)
        V_peak = A_max * t_accel
        t_flat = 0
        print(f"  Triangle Profile: t_accel = {t_accel:.2f}s, V_peak = {V_peak:.2f} deg/s")
    else:
        # Trapezoidal profile
        D_flat = D - 2 * D_accel
        t_flat = D_flat / V_max
        V_peak = V_max
        print(f"  Trapezoidal Profile: t_accel = {t_accel:.2f}s, t_flat = {t_flat:.2f}s, V_peak = {V_peak:.2f} deg/s")
    
    T = 2 * t_accel + t_flat
    q = np.zeros_like(t)
    
    for idx, ti in enumerate(t):
        if ti < t_accel:
            # Acceleration phase
            q[idx] = 0.5 * A_max * ti**2
        elif ti < t_accel + t_flat:
            # Cruise phase
            q[idx] = D_accel + V_peak * (ti - t_accel)
        elif ti <= T:
            # Deceleration phase
            td = T - ti
            q[idx] = D - 0.5 * A_max * td**2
        else:
            q[idx] = D
    return q * direction

for i in range(len(initial)):
    q0 = initial[i]
    qf = final[i]
    print(f"Joint {i+1}:")
    joint_trajectory_trapezoidal[:, i] = trapezoidal_profile(q0, qf, V_max[i], A_max[i], tf, t)
    
    # Maximum acceleration is A_max
    print(f"  Maximum Acceleration: {A_max[i]} deg/s²\n")

print("\n")

# -------------------- Plotting All Trajectories --------------------
plt.figure(figsize=(18, 12))

methods = {
    'Linear Interpolation': joint_trajectory_linear,
    'Cubic Polynomial': joint_trajectory_cubic,
    'Quintic Polynomial': joint_trajectory_quintic,
    'B-Spline (Quadratic)': joint_trajectory_bspline,
    'Bezier Curve (Linear)': joint_trajectory_bezier,
    'Trapezoidal Velocity': joint_trajectory_trapezoidal
}

for idx, (method, trajectory) in enumerate(methods.items(), 1):
    plt.subplot(3, 2, idx)
    for j in range(trajectory.shape[1]):
        plt.plot(t, trajectory[:, j], label=f'Joint {j+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angles (degrees)')
    plt.title(f'Joint Trajectory using {method}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------- Dynamic Simulation on PyBullet --------------------
# Function to get joint indices (assuming revolute joints and ignoring fixed joints)
def get_revolute_joint_indices(robot):
    num_joints = p.getNumJoints(robot)
    revolute_joints = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot, i)
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE:
            revolute_joints.append(i)
    return revolute_joints

# Get revolute joint indices
revolute_joints = get_revolute_joint_indices(robot_id)
print(f"Revolute Joint Indices: {revolute_joints}\n")

# Function to set joint angles
def set_joint_angles(robot, joint_indices, angles):
    for idx, joint in enumerate(joint_indices):
        p.setJointMotorControl2(bodyIndex=robot,
                                jointIndex=joint,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=m.radians(angles[idx]),
                                force=500)

# Function to simulate a trajectory
def simulate_trajectory(robot, joint_indices, trajectory, tf, num_points):
    slowdown_factor = 2  # Adjust this value to slow down the simulation
    for point in range(num_points):
        angles = trajectory[point]
        set_joint_angles(robot, joint_indices, angles)
        p.stepSimulation()
        time.sleep((tf / num_points) * slowdown_factor)  # Sync simulation with real time

# -------------------- Simulate Each Trajectory Method --------------------
print("----- Starting Dynamic Simulation -----\n")

# List of trajectories to simulate
trajectory_methods = {
    'Linear Interpolation': joint_trajectory_linear,
    'Cubic Polynomial': joint_trajectory_cubic,
    'Quintic Polynomial': joint_trajectory_quintic,
    'B-Spline (Quadratic)': joint_trajectory_bspline,
    'Bezier Curve (Linear)': joint_trajectory_bezier,
    'Trapezoidal Velocity': joint_trajectory_trapezoidal
}

for method_name, trajectory in trajectory_methods.items():
    print(f"Simulating {method_name}...")
    simulate_trajectory(robot_id, revolute_joints, trajectory, tf, num_points)
    print(f"{method_name} simulation completed.\n")
    time.sleep(1)  # Pause between methods

# Disconnect from PyBullet
p.disconnect()
