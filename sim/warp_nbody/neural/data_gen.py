import dataclasses
import pathlib

import numpy as np
import h5py
import warp as wp

from ..kernels.physics import kernel_forces, kernel_integrate
from ..spawner import spawn_sphere


@dataclasses.dataclass
class DataGenConfig:
    N_PARTICLES: int = 1000
    N_EPISODES: int = 200
    N_STEPS: int = 500
    DT: float = 0.01
    G: float = 1.0
    SOFTENING: float = 0.05
    OUTPUT_PATH: str = str(pathlib.Path(__file__).resolve().parents[3] / "data" / "nbody_dataset.h5")


def generate_dataset(config: DataGenConfig) -> None:
    wp.init()

    n = config.N_PARTICLES
    total_frames = config.N_EPISODES * config.N_STEPS

    all_positions = np.empty((total_frames, n, 3), dtype=np.float32)
    all_velocities = np.empty((total_frames, n, 3), dtype=np.float32)
    all_masses = np.empty((total_frames, n, 1), dtype=np.float32)
    all_accelerations = np.empty((total_frames, n, 3), dtype=np.float32)

    softening_sq = config.SOFTENING ** 2
    frame_idx = 0

    for episode in range(config.N_EPISODES):
        positions_np, velocities_np, masses_np = spawn_sphere(
            n, radius=50.0, body_mass=1.0, speed_scale=0.5,
        )

        rng = np.random.default_rng(seed=episode)
        positions_np += rng.normal(0, 1.0, positions_np.shape).astype(np.float32)
        velocities_np += rng.normal(0, 0.1, velocities_np.shape).astype(np.float32)

        pos_wp = wp.array(positions_np, dtype=wp.vec3, device="cuda")
        vel_wp = wp.array(velocities_np, dtype=wp.vec3, device="cuda")
        mass_wp = wp.array(masses_np, dtype=float, device="cuda")
        forces_wp = wp.zeros(n, dtype=wp.vec3, device="cuda")
        active_wp = wp.ones(n, dtype=int, device="cuda")

        for step in range(config.N_STEPS):
            wp.launch(kernel_forces, dim=n, device="cuda", inputs=[
                pos_wp, mass_wp, active_wp, forces_wp,
                config.G, softening_sq, n,
            ])

            pos_t = wp.to_torch(pos_wp).cpu().numpy()
            vel_t = wp.to_torch(vel_wp).cpu().numpy()
            forces_t = wp.to_torch(forces_wp).cpu().numpy()
            mass_np = wp.to_torch(mass_wp).cpu().numpy()

            acc_t = forces_t / mass_np[:, np.newaxis]

            all_positions[frame_idx] = pos_t
            all_velocities[frame_idx] = vel_t
            all_masses[frame_idx] = mass_np[:, np.newaxis]
            all_accelerations[frame_idx] = acc_t

            wp.launch(kernel_integrate, dim=n, device="cuda", inputs=[
                pos_wp, vel_wp, forces_wp, mass_wp, active_wp, config.DT,
            ])

            frame_idx += 1

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{config.N_EPISODES} done")

    output_path = pathlib.Path(config.OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_path), "w") as f:
        f.create_dataset("positions", data=all_positions, compression="gzip")
        f.create_dataset("velocities", data=all_velocities, compression="gzip")
        f.create_dataset("masses", data=all_masses, compression="gzip")
        f.create_dataset("accelerations", data=all_accelerations, compression="gzip")

    print(f"Dataset saved to {output_path} — {total_frames} frames, {n} particles each")


if __name__ == "__main__":
    cfg = DataGenConfig()
    print(f"Generating dataset: {cfg.N_EPISODES} episodes x {cfg.N_STEPS} steps x {cfg.N_PARTICLES} particles")
    generate_dataset(cfg)
