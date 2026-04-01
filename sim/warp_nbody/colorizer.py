import numpy as np
import warp as wp


# colorize based on mass and velocity
@wp.kernel
def kernel_colorize(
    masses:     wp.array(dtype=float),
    velocities: wp.array(dtype=wp.vec3),
    active:     wp.array(dtype=int),
    colors:     wp.array(dtype=wp.vec3),
    max_mass:   float,
    max_speed:  float,
):
    i = wp.tid()
    if active[i] == 0:
        colors[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    t_mass  = wp.min(masses[i] / max_mass, 1.0)
    t_speed = wp.min(wp.length(velocities[i]) / max_speed, 1.0)
    t = t_mass * 0.7 + t_speed * 0.3
    if t < 0.5:
        s = t * 2.0
        colors[i] = wp.vec3(s * 0.9 + 0.1, s * 0.9 + 0.1, 1.0)
    else:
        s = (t - 0.5) * 2.0
        colors[i] = wp.vec3(1.0, 1.0 - s * 0.6, 1.0 - s)


class ColorManager:

    def __init__(self):
        self._colors_wp = None

    def allocate(self, n: int) -> None:
        self._colors_wp = wp.zeros(n, dtype=wp.vec3, device="cuda")

    def get_colors(self, sim, masses_np: np.ndarray, velocities_np: np.ndarray) -> np.ndarray:
        speeds_np = np.linalg.norm(velocities_np, axis=1)
        max_mass  = float(masses_np.max()) or 1.0
        max_speed = float(speeds_np.max()) or 1.0
        wp.launch(kernel_colorize, dim=sim._n, device="cuda", inputs=[
            sim.masses, sim.velocities, sim.active,
            self._colors_wp, max_mass, max_speed,
        ])
        return self._colors_wp.numpy()

    def free(self) -> None:
        self._colors_wp = None
