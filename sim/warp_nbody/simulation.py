import numpy as np
import warp as wp

BASE_MASS:   float = 1.0
BASE_RADIUS: float = 0.3

# positions, masses, forces -> compute forces
@wp.kernel
def kernel_forces(
    positions:   wp.array(dtype=wp.vec3),
    masses:      wp.array(dtype=float),
    active:      wp.array(dtype=int),
    forces:      wp.array(dtype=wp.vec3),
    G:           float,
    softening_sq: float,
    n:           int,
):
    i = wp.tid()
    if active[i] == 0:
        return
    f  = wp.vec3(0.0, 0.0, 0.0)
    pi = positions[i]
    mi = masses[i]
    for j in range(n):
        if j == i or active[j] == 0:
            continue
        r        = positions[j] - pi
        dist_sq  = wp.dot(r, r) + softening_sq
        inv_dist3 = 1.0 / (dist_sq * wp.sqrt(dist_sq))
        f = f + r * (G * mi * masses[j] * inv_dist3)
    forces[i] = f


# positions, velocities, forces, masses -> compute new velocities and positions
@wp.kernel
def kernel_integrate(
    positions:  wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces:     wp.array(dtype=wp.vec3),
    masses:     wp.array(dtype=float),
    active:     wp.array(dtype=int),
    dt:         float,
):
    i = wp.tid()
    if active[i] == 0:
        return
    acc         = forces[i] * (1.0 / masses[i])
    velocities[i] = velocities[i] + acc * dt
    positions[i]  = positions[i]  + velocities[i] * dt



# 2 passes to account for object merging
# pass 1: Check which objects get merged by which objects (if merge_into[i] = -1 -> Remains, if merge_into[i] = j, i is eaten by j)
# pass 2: Compute new radius for objects that ate other objects
@wp.kernel
def kernel_accrete_pass1(
    positions:  wp.array(dtype=wp.vec3),
    masses:     wp.array(dtype=float),
    radii:      wp.array(dtype=float),
    active:     wp.array(dtype=int),
    merge_into: wp.array(dtype=int),
    n:          int,
):
    i = wp.tid()
    if active[i] == 0:
        return
    merge_into[i] = -1
    pi = positions[i]
    mi = masses[i]
    for j in range(n):
        if j == i or active[j] == 0:
            continue
        dist = wp.length(positions[j] - pi)
        if dist < radii[i] + radii[j]:
            if masses[j] > mi or (masses[j] == mi and j < i):
                merge_into[i] = j
                return
            

@wp.kernel # TODO:  look into a warp.fill function
def kernel_reset_int(arr: wp.array(dtype=int)):
    arr[0] = int(0)


@wp.kernel #TODO: Check tile sum or some sort of reduction tree in WARP
def kernel_count_active(active: wp.array(dtype=int), count: wp.array(dtype=int)):
    i = wp.tid()
    if active[i] != 0:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def kernel_accrete_pass2(
    masses:     wp.array(dtype=float),
    radii:      wp.array(dtype=float),
    active:     wp.array(dtype=int),
    merge_into: wp.array(dtype=int),
    base_mass:   float,
    base_radius: float,
):
    i = wp.tid()
    if active[i] == 0 or merge_into[i] == -1:
        return
    j = merge_into[i]
    wp.atomic_add(masses, j, masses[i])
    active[i] = 0
    radii[j]  = base_radius * wp.pow(masses[j] / base_mass, 1.0 / 3.0)


class NBodySimulation:

    def __init__(self):
        self.positions:  wp.array | None = None
        self.velocities: wp.array | None = None
        self.masses:     wp.array | None = None
        self.radii:      wp.array | None = None
        self.active:     wp.array | None = None
        self.forces:     wp.array | None = None

        self._n:            int = 0
        self._frame:        int = 0
        self._active_count: wp.array | None = None

        self.G:                  float = 0.001
        self.softening:          float = 0.05
        self.dt:                 float = 0.01
        self.accretion_enabled:  bool  = True
        self.accretion_interval: int   = 5

    def allocate(self, positions_np, velocities_np, masses_np) -> None:
        n        = len(masses_np)
        self._n  = n
        self._frame = 0

        self.positions  = wp.array(positions_np,  dtype=wp.vec3, device="cuda")
        self.velocities = wp.array(velocities_np, dtype=wp.vec3, device="cuda")
        self.masses     = wp.array(masses_np,     dtype=float,   device="cuda")
        self.forces     = wp.zeros(n, dtype=wp.vec3, device="cuda")
        self.active     = wp.ones(n,  dtype=int,   device="cuda")

        radii_np   = (BASE_RADIUS * (masses_np / BASE_MASS) ** (1.0 / 3.0)).astype(np.float32)
        self.radii = wp.array(radii_np, dtype=float, device="cuda")

        self._active_count = wp.zeros(1, dtype=int, device="cuda")

    def free(self) -> None:
        self.positions     = None
        self.velocities    = None
        self.masses        = None
        self.radii         = None
        self.active        = None
        self.forces        = None
        self._active_count = None
        self._n            = 0
        self._frame        = 0

    def count_active(self) -> int:
        # returns the number of active bodies. copies only 1 int from GPU -> CPU
        if self.active is None:
            return 0
        wp.launch(kernel_reset_int,    dim=1,       device="cuda", inputs=[self._active_count])
        wp.launch(kernel_count_active, dim=self._n, device="cuda", inputs=[
            self.active, self._active_count,
        ])
        return int(self._active_count.numpy()[0])

    def step(self) -> None:
        if self.positions is None:
            return
        self._run_forces()
        self._run_integrate()
        if self.accretion_enabled and self._frame % self.accretion_interval == 0:
            self._run_accrete()
        self._frame += 1

    def _run_forces(self) -> None:
        wp.launch(kernel_forces, dim=self._n, device="cuda", inputs=[
            self.positions, self.masses, self.active, self.forces,
            self.G, self.softening ** 2, self._n,
        ])

    def _run_integrate(self) -> None:
        wp.launch(kernel_integrate, dim=self._n, device="cuda", inputs=[
            self.positions, self.velocities, self.forces, self.masses, self.active, self.dt,
        ])

    def _run_accrete(self) -> None:
        merge_into = wp.full(self._n, -1, dtype=int, device="cuda")
        wp.launch(kernel_accrete_pass1, dim=self._n, device="cuda", inputs=[
            self.positions, self.masses, self.radii, self.active, merge_into, self._n,
        ])
        wp.launch(kernel_accrete_pass2, dim=self._n, device="cuda", inputs=[
            self.masses, self.radii, self.active, merge_into, BASE_MASS, BASE_RADIUS,
        ])
