import numpy as np
import carb
import omni.usd
from pxr import Vt, UsdGeom

from .instancer import INSTANCER_PATH

VISUAL_SCALE = 3.0
VISUAL_CAP   = 15.0


def _to_vec3f(arr: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray.FromNumpy(
        np.ascontiguousarray(arr, dtype=np.float32).reshape(-1, 3)
    )


class FabricBridge:

    def __init__(self):
        self._sim       = None
        self._n         = 0
        self._colorizer = None
        self._instancer = None

    def bind(self, sim, n_bodies: int, colorizer) -> None:
        self._sim       = sim
        self._n         = n_bodies
        self._colorizer = colorizer

        stage = omni.usd.get_context().get_stage()
        self._instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(INSTANCER_PATH))

    def mark_dirty(self) -> None:
        if self._sim is None:
            return

        # copy the arrays to CPU (TODO: Keep everything on Fabric GPU API)
        positions_np  = np.ascontiguousarray(self._sim.positions.numpy(),  dtype=np.float32).reshape(self._n, 3)
        masses_np     = np.ascontiguousarray(self._sim.masses.numpy(),     dtype=np.float32).reshape(self._n)
        velocities_np = np.ascontiguousarray(self._sim.velocities.numpy(), dtype=np.float32).reshape(self._n, 3)
        radii_np      = np.ascontiguousarray(self._sim.radii.numpy(),      dtype=np.float32).reshape(self._n)
        active_np     = np.ascontiguousarray(self._sim.active.numpy(),     dtype=np.int32  ).reshape(self._n)

        # compute visual sizes
        mask     = active_np.astype(bool)
        visual_r = np.minimum(radii_np[mask] * VISUAL_SCALE, VISUAL_CAP).astype(np.float32)

        scales_np               = np.zeros((self._n, 3), dtype=np.float32)
        scales_np[mask, 0]      = visual_r
        scales_np[mask, 1]      = visual_r
        scales_np[mask, 2]      = visual_r

        # copy color array to CPU (TODO: Keep everything on Fabric GPU API)
        colors_np = np.ascontiguousarray(
            self._colorizer.get_colors(self._sim, masses_np, velocities_np), dtype=np.float32
        ).reshape(self._n, 3)

        self._instancer.GetPositionsAttr().Set(_to_vec3f(positions_np))
        self._instancer.GetScalesAttr().Set(_to_vec3f(scales_np))
        self._instancer.GetPrim().GetAttribute("primvars:displayColor").Set(_to_vec3f(colors_np))

    def unbind(self) -> None:
        self._sim       = None
        self._n         = 0
        self._colorizer = None
        self._instancer = None
