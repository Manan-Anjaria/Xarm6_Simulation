# XArm6 Robot Control and Simulation

This repository contains a Python implementation for controlling and simulating an XArm6 robot, including forward/inverse kinematics, trajectory planning, and visualization.

## Prerequisites

### Required Python Libraries
```bash
pip install numpy
pip install matplotlib
pip install pybullet
```

### PyBullet Data and URDF Files
The code requires the XArm6 URDF file for simulation. The files are included in the PyBullet data package. By default, the code looks for the URDF at:
```
/bullet3-master/examples/pybullet/gym/pybullet_data/xarm/xarm6_robot.urdf
```

To configure the URDF path:
1. Locate your PyBullet installation directory(included in the code package)
2. Find the `pybullet_data` folder
3. Update the `urdf_path` parameter in the `visualize_simulation` function'
4. Modify the URDF path in the code:(Line 546,1114)
```python
def visualize_simulation(..., urdf_path="path/to/your/xarm6_robot.urdf"):
```

## Code Structure

### Main Components
- `XArm6Robot` class: Core robot control implementation
- Matrix operations
- Kinematics computations
- Trajectory planning
- Visualization tools

### Key Features
- Forward and inverse kinematics
- Workspace analysis
- Multiple trajectory generation methods
- PyBullet simulation
- Visualization plots

## Usage

### Basic Usage
```python
# Define joint limits(Line 617)
joint_limits_deg = [
    (-170, 170),  # Joint 1
    (-120, 120),  # Joint 2
    (-170, 170),  # Joint 3
    (-120, 120),  # Joint 4
    (-170, 170),  # Joint 5
    (-120, 120)   # Joint 6
]

# Define initial joint angles(line 627)
initial_angles = [100, -10, -110, -50, -50, 50]

#Waypoint Generation(Line 664)
    # **Define 7 waypoints for the end effector to reach**
    current_end_effector_pos = robot.get_end_effector_position()
    waypoints = [
        [current_end_effector_pos[0] + 260, current_end_effector_pos[1], current_end_effector_pos[2]],    # Move +X
        [current_end_effector_pos[0], current_end_effector_pos[1] + 230, current_end_effector_pos[2]],    # Move +Y
        [current_end_effector_pos[0], current_end_effector_pos[1], current_end_effector_pos[2] + 60],    # Move +Z
        [current_end_effector_pos[0] - 190, current_end_effector_pos[1], current_end_effector_pos[2]],   # Move -X
        [current_end_effector_pos[0], current_end_effector_pos[1] - 270, current_end_effector_pos[2]],   # Move -Y
        [current_end_effector_pos[0], current_end_effector_pos[1], current_end_effector_pos[2] - 60],   # Move -Z
        [current_end_effector_pos[0] + 140, current_end_effector_pos[1] + 60, current_end_effector_pos[2] + 80]   # Move diagonally
    ]
## Configuration


### Simulation Parameters
Modify simulation settings:
- `simulation_duration`: Duration in seconds
- `simulation_steps`: Number of steps
- `slowdown_factor`: Simulation speed factor

## Troubleshooting

### Common Issues
1. URDF File Not Found
   - Verify PyBullet installation
   - Check URDF path
   - Ensure file permissions

2. Simulation Issues
   - Check joint limits
   - Verify PyBullet connection
   - Monitor simulation parameters

3. Convergence Problems
   - Adjust damping factor
   - Increase maximum iterations(Once my attempt converged on 9921 iteration)
   - Check target position feasibility

## Notes
- The code implements matrix operations without external libraries
- Joint angles are in degrees for input/output
- Internal calculations use radians
- PyBullet simulation runs at 240Hz by default

## Author
Manan Anjaria (mha9531)  
MS Mechatronics and Robotics  
New York University