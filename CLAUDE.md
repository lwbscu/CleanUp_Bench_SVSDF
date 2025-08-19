# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is REMANI-Planner adapted for IsaacSim 4.5, a real-time whole-body motion planning system for mobile manipulators. The project was originally built for RViz simulation and has been migrated to use IsaacSim 4.5 with IsaacLab for more realistic physics simulation.

**Key Architecture Components:**
- **Motion Planning Core**: Original REMANI planner algorithms using environment-adaptive search and spatial-temporal optimization
- **IsaacSim Interface**: Replacement for RViz-based simulation using IsaacSim 4.5 and IsaacLab
- **Mobile Manipulator Control**: Handles iRobot Create 3 mobile base with robotic arm
- **Real-time Planning**: FSM-based replanning system for dynamic environments

## Environment Setup and Running

### Prerequisites
- IsaacSim 4.5 installed at `/home/lwb/isaacsim/`
- Conda environment `isaaclab_4_5_0` 
- Robot model: `/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd`

## Robot Platform Specifications

### iRobot Create 3 Mobile Base
- **Mobility**: Differential drive mobile platform
- **Sensors**: IMU, wheel encoders, cliff sensors, IR sensors
- **Payload**: ~9kg capacity for manipulation payload
- **Communication**: ROS 2 native interface
- **Power**: Rechargeable battery with docking capability

### Franka Emika Panda Arm (Integrated)
- **DOF**: 7-axis collaborative robot arm
- **Payload**: 3kg at full extension
- **Reach**: 855mm working radius
- **Control**: Joint torque control with force sensing
- **Safety**: Built-in collision detection and safe operation

## Data Flow Architecture

### Core Data Structures
```
State Space:
- Mobile Base: [x, y, θ] ∈ SE(2)
- Manipulator: [q₁...q₇] ∈ R⁷ (joint angles)
- Combined: SE(2) × R⁷ configuration space

Sensor Data Pipeline:
Environment → Perception → Planning → Control → Actuation
```

### Planning Data Flow
1. **Perception Layer**: Point cloud → Occupancy grid → Collision map
2. **Planning Core**: Current state → REMANI planner → Trajectory waypoints  
3. **Control Layer**: Waypoints → Joint commands → Motor control
4. **Feedback Loop**: Sensor feedback → State estimation → Replanning

### Critical Design Principles
- **Zero abstraction tolerance**: Direct sensor-to-action mapping
- **Stateful planning**: Maintain planning context, discard version layers
- **Real-time constraints**: <100ms planning cycles, no buffering overhead
- **Monolithic core**: Single planning thread, eliminate inter-process communication

## Code Review Philosophy

### Problem Identification Strategy
- **Root cause analysis**: Trace issues to fundamental design flaws, not symptoms
- **Compatibility layer elimination**: Remove version compatibility shims that add complexity
- **Direct problem solving**: Fix core algorithms, not wrapper functions

### Anti-Patterns to Eliminate
- Multiple abstraction layers for simple operations
- "Future-proofing" for hypothetical requirements
- Defensive coding that masks real issues
- Bridge patterns between similar interfaces

### Debugging Approach
1. **Data flow tracing**: Follow data from sensor input to actuator output
2. **State inspection**: Verify configuration space consistency
3. **Timing analysis**: Measure actual vs. required planning cycles
4. **Interface reduction**: Minimize APIs, maximize direct function calls

### Running the System

**Integrated REMANI-Isaac Sim Interface (recommended):**
```bash
./run_remani_integrated.sh
```

**Manual IsaacSim setup:**
```bash
cd ~/isaacsim
conda activate isaaclab_4_5_0
source setup_conda_env.sh
python /path/to/remani_isaac_integrated.py
```

## Integration Architecture

### Original REMANI Components Preserved
- **Mobile Manipulator Planning**: SE(2) × R⁷ configuration space planning
- **Waypoint Navigation**: Multi-target trajectory optimization
- **FSM-based Replanning**: Real-time adaptive replanning system
- **Joint Space Control**: 7-DOF arm configuration management

### Isaac Sim 4.5 Integration Layer
- **Physics Simulation**: Replaces RViz with realistic physics
- **Visual Rendering**: Real-time 3D visualization 
- **Robot Model**: Create3 + Franka arm USD integration
- **Scene Management**: Dynamic obstacle and environment handling

### Data Flow Integration
```
YAML Config → REMANIConfig → SimplifiedREMANIPlanner
     ↓                              ↓
Waypoint Targets ←→ Mobile Base Commands ←→ Isaac Sim Physics
     ↓                              ↓
Joint Commands ←→ Robot State Updates ←→ Visual Feedback
```

