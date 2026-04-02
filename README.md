# sim.warp_nbody

GPU-accelerated N-body gravitational simulation running inside NVIDIA Omniverse, powered by [WARP](https://github.com/NVIDIA/warp).

![demo](sim_trimmed.gif)

## How it works

All physics runs on the GPU using WARP kernels. Every frame, gravity is computed between all pairs of bodies (O(N^2)), velocities and positions are integrated, and overlapping bodies merge together. The goal is for nothing to leave the GPU.

## Neural Force Field

A GNN (Graph Neural Network) can approximate the N-body forces and run side-by-side with the classical simulation for comparison.

**Pipeline:**
1. **Generate data** - runs the classical simulation and records positions, velocities, masses, and accelerations to HDF5
2. **Train** - trains a GNS-style GNN (3 message-passing rounds, radius graph) on the recorded data
3. **Inference** - the trained model predicts forces via zero-copy Warp-to-PyTorch bridge

**Architecture:** Node encoder (vel + mass) and edge encoder (relative pos + distance + relative vel) feed into 3 message-passing layers with residual connections, decoded to per-particle accelerations (simplest format I could find online).

**Dual-stream mode:** When enabled, both classical (blue) and neural (orange) particles spawn from identical initial conditions. The position error between them is logged every 100 frames.

**Tunable parameters:**
- **Cutoff radius** - controls the radius graph size (smaller = faster, less accurate)
- **Inference interval** - run the GNN every K frames and reuse cached forces in between

## Presets

- **Galaxy Disk** - disk of bodies orbiting a central mass
- **Sphere** - random uniform sphere
- **Solar System** - star, 8 planets, and an asteroid belt
- **Random** - random box of bodies
- **Binary Galaxy** - two colliding galaxy disks
- **Black Hole** - dense inner ring with outer spiral drift

## Project Structure

```
sim/warp_nbody/
  extension.py          Omniverse Kit extension entry point
  simulation.py         main simulation loop (classical + neural dual-stream)
  fabric_bridge.py      USDRT/Fabric zero-copy GPU<>USD sync
  instancer.py          USD particle instancers (classical + neural)
  spawner.py            initial condition generators
  colorizer.py          per-particle color assignment
  kernels/
    physics.py          Warp N-body force + integration kernels
    visual.py           Warp kernels for color/scale updates
  neural/
    model.py            GNS-style GNN (PyTorch Geometric)
    inference.py        Warp<->PyTorch zero-copy bridge
    data_gen.py         HDF5 training data generation
    train.py            training loop
  ui/
    panel.py            omni.ui panel (simulation + neural controls)
```

## Requirements

- NVIDIA Omniverse Kit
- NVIDIA WARP
- CUDA GPU
- PyTorch (CUDA build), PyTorch Geometric, torch_cluster, h5py (for neural features)

## Profiling with Nsight

```
nsys launch \
  --trace=cuda,nvtx \
  .../kit-app-template/_build/linux-x86_64/release/kit/kit \
  .../kit-app-template/_build/linux-x86_64/release/apps/my_company.my_usd_composer.kit

# Once the sim is running:
nsys start
# wait a few seconds...
nsys stop

QT_QPA_PLATFORM=xcb nsys-ui .../report1.nsys-rep
```
