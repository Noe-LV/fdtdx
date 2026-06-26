import pathlib

import gdstk
import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.grid import polygon_to_mask, polygon_to_mask_at_points
from fdtdx.core.jax.pytrees import autoinit, field, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject


@autoinit
class ExtrudedPolygon(StaticMultiMaterialObject):
    """A polygon object specified by a list of vertices.

    The vertices must be given in a coordinate system centered at the origin, i.e. (0, 0)
    corresponds to the center of the object's bounding box. The polygon is placed so that
    its center coincides with the center of the grid region allocated to this object.

    The cross-section size is automatically inferred from the vertex bounding box for the
    two axes perpendicular to ``axis``, so ``partial_real_shape`` does not need to be
    specified for those axes.  The extrusion axis size must still be determined by a
    constraint or an explicit ``partial_real_shape`` entry.
    """

    #: Name of the material in the materials dictionary to be used for the object
    material_name: str = frozen_field()

    #: The extrusion axis.
    axis: int = frozen_field()

    #: numpy array of shape (N, 2) with vertices in metrical units (meter), centered at origin.
    vertices: np.ndarray = frozen_field()

    def __post_init__(self):
        w = float(self.vertices[:, 0].max() - self.vertices[:, 0].min())
        h = float(self.vertices[:, 1].max() - self.vertices[:, 1].min())
        real_shape = list(self.partial_real_shape)
        grid_shape = list(self.partial_grid_shape)
        for ax, size in ((self.horizontal_axis, w), (self.vertical_axis, h)):
            if real_shape[ax] is not None:
                raise Exception(
                    f"ExtrudedPolygon {self.name}: partial_real_shape for axis {ax} is derived from the "
                    f"vertex bounding box ({size:.3e} m). Do not specify it explicitly."
                )
            if grid_shape[ax] is not None:
                raise Exception(
                    f"ExtrudedPolygon {self.name}: partial_grid_shape for axis {ax} is derived from the "
                    f"vertex bounding box. Do not specify it explicitly."
                )
            real_shape[ax] = size
        object.__setattr__(self, "partial_real_shape", tuple(real_shape))

    @property
    def horizontal_axis(self) -> int:
        """Gets the horizontal axis perpendicular to the fiber axis."""
        return get_transverse_axes(self.axis)[0]

    @property
    def vertical_axis(self) -> int:
        """Gets the vertical axis perpendicular to the fiber axis."""
        return get_transverse_axes(self.axis)[1]

    def get_voxel_mask_for_shape(self) -> jax.Array:
        n_horizontal = self.grid_shape[self.horizontal_axis]
        n_vertical = self.grid_shape[self.vertical_axis]

        # Shift vertices from object-center coords to local grid coords.
        center_h = 0.5 * self.real_shape[self.horizontal_axis]
        center_v = 0.5 * self.real_shape[self.vertical_axis]
        grid_vertices = self.vertices + np.array([center_h, center_v])

        grid = self._config.resolved_grid
        if grid is None:
            spacing = self._config.uniform_spacing()
            half_res = 0.5 * spacing
            max_horizontal = (n_horizontal - 0.5) * spacing
            max_vertical = (n_vertical - 0.5) * spacing

            mask_2d = polygon_to_mask(
                boundary=(half_res, half_res, max_horizontal, max_vertical),
                resolution=spacing,
                polygon_vertices=grid_vertices,
            )
        else:
            h_lower, h_upper = self.grid_slice_tuple[self.horizontal_axis]
            v_lower, v_upper = self.grid_slice_tuple[self.vertical_axis]
            h_edges = np.asarray(grid.edges(self.horizontal_axis))
            v_edges = np.asarray(grid.edges(self.vertical_axis))
            h_centers = 0.5 * (h_edges[h_lower:h_upper] + h_edges[h_lower + 1 : h_upper + 1]) - h_edges[h_lower]
            v_centers = 0.5 * (v_edges[v_lower:v_upper] + v_edges[v_lower + 1 : v_upper + 1]) - v_edges[v_lower]
            mask_2d = polygon_to_mask_at_points(
                x_coords=h_centers,
                y_coords=v_centers,
                polygon_vertices=grid_vertices,
            )
        extrusion_height = self.grid_shape[self.axis]
        mask = jnp.repeat(
            jnp.expand_dims(jnp.asarray(mask_2d, dtype=jnp.bool), axis=self.axis),
            repeats=extrusion_height,
            axis=self.axis,
        )

        return mask

    def get_material_mapping(
        self,
    ) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        arr = jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
        return arr


def extruded_polygon_from_gds(
    lib: gdstk.Library,
    cell_name: str,
    layer: int,
    datatype: int = 0,
    polygon_index: int = 0,
    **kwargs,
) -> ExtrudedPolygon:
    """Create an ExtrudedPolygon from a polygon in an already-loaded gdstk Library.

    Args:
        lib: An already-loaded gdstk Library.
        cell_name: Name of the GDS cell containing the polygon.
        layer: GDS layer number to read.
        datatype: GDS datatype (default 0).
        polygon_index: Which polygon to use when multiple exist on the layer (default 0).
        **kwargs: Forwarded to ExtrudedPolygon (axis, material_name, materials, …).

    Returns:
        ExtrudedPolygon with vertices centered around the origin in metres.

    Raises:
        ValueError: If the cell or layer/datatype combination is not found.
        IndexError: If polygon_index is out of range.
    """
    cell = next((c for c in lib.cells if isinstance(c, gdstk.Cell) and c.name == cell_name), None)
    if cell is None:
        raise ValueError(f"Cell '{cell_name}' not found in library")

    matching = [p for p in cell.polygons if p.layer == layer and p.datatype == datatype]
    if not matching:
        raise ValueError(f"No polygons on layer={layer}, datatype={datatype} in cell '{cell_name}'")
    if polygon_index >= len(matching):
        raise IndexError(
            f"polygon_index={polygon_index} out of range; found {len(matching)} polygon(s) on layer={layer}"
        )

    polygon = matching[polygon_index]
    vertices_m = np.array(polygon.points) * lib.unit  # library units → metres

    # centre vertices around origin (ExtrudedPolygon convention)
    centre = 0.5 * (vertices_m.min(axis=0) + vertices_m.max(axis=0))
    centred = vertices_m - centre

    return ExtrudedPolygon(vertices=centred, **kwargs)


def extruded_polygon_from_gds_path(
    gds_file: str | pathlib.Path,
    cell_name: str,
    layer: int,
    datatype: int = 0,
    polygon_index: int = 0,
    **kwargs,
) -> ExtrudedPolygon:
    """Create an ExtrudedPolygon from a polygon in a GDS file.

    Args:
        gds_file: Path to the .gds file.
        cell_name: Name of the GDS cell containing the polygon.
        layer: GDS layer number to read.
        datatype: GDS datatype (default 0).
        polygon_index: Which polygon to use when multiple exist on the layer (default 0).
        **kwargs: Forwarded to ExtrudedPolygon (axis, material_name, materials, …).

    Returns:
        ExtrudedPolygon with vertices centered around the origin in metres.

    Raises:
        ValueError: If the cell or layer/datatype combination is not found.
        IndexError: If polygon_index is out of range.
    """
    lib = gdstk.read_gds(str(gds_file))
    return extruded_polygon_from_gds(lib, cell_name, layer, datatype, polygon_index, **kwargs)


@autoinit
class DifferentiableExtrudedPolygon(StaticMultiMaterialObject):
    """An extruded polygon for differentiable shape optimization.

    Unlike the static ``ExtrudedPolygon``, the vertex array is a regular
    (non-frozen) JAX field so gradients can flow through vertex positions
    during inverse design. The mask returned by ``get_voxel_mask_for_shape``
    is a soft float array in ``[0, 1]`` computed via a smooth polygon signed
    distance field (SDF), making it fully differentiable w.r.t. ``vertices``.

    After placement constraints are solved and before any jit-compiled
    function is called, ``finalize()`` must be called on the object (or on
    the container via ``finalize_differentiable_polygons``). This extracts
    all static grid geometry into plain numpy arrays so nothing inside
    ``get_voxel_mask_for_shape`` needs to touch the config pytree under jit.

    Args:
        material_name: Name of the material in the ``materials`` dict.
        axis: Extrusion axis (0, 1, or 2).
        vertices: ``(N, 2)`` array of polygon vertices in metrical units
            (meters), centered at the origin. Non-frozen — participates in
            JAX differentiation.
        smoothing_width: Width of the SDF transition band in meters.
            ``None`` (default) uses one grid cell width.
    """

    # ------------------------------------------------------------------ #
    # Public constructor fields                                            #
    # ------------------------------------------------------------------ #

    material_name: str = frozen_field()
    axis: int = frozen_field()

    #: Non-frozen — the only traced leaf. Shape (N, 2), meters, centered.
    vertices: jax.Array = field()

    smoothing_width: float | None = frozen_field(default=None)

    # ------------------------------------------------------------------ #
    # Private fields set in __post_init__ (pre-placement)                 #
    # ------------------------------------------------------------------ #

    #: Static vertex count — never read verts.shape[0] under jit.
    _n_vertices: int = frozen_field(default=0, init=False)

    # ------------------------------------------------------------------ #
    # Private fields set in finalize() (post-placement)                   #
    # These hold plain numpy arrays extracted from the config before jit. #
    # ------------------------------------------------------------------ #

    _h_centers_np: np.ndarray | None = frozen_field(default=None, init=False)
    _v_centers_np: np.ndarray | None = frozen_field(default=None, init=False)
    _center_h: float = frozen_field(default=0.0, init=False)
    _center_v: float = frozen_field(default=0.0, init=False)
    _smoothing_hw: float = frozen_field(default=0.0, init=False)
    _finalized: bool = frozen_field(default=False, init=False)

    # ------------------------------------------------------------------ #
    # __post_init__: only what is available before placement              #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        verts_np = np.asarray(self.vertices)

        w = float(verts_np[:, 0].max() - verts_np[:, 0].min())
        h = float(verts_np[:, 1].max() - verts_np[:, 1].min())
        real_shape = list(self.partial_real_shape)
        grid_shape = list(self.partial_grid_shape)
        for ax, size in ((self.horizontal_axis, w), (self.vertical_axis, h)):
            if real_shape[ax] is not None:
                raise ValueError(
                    f"DifferentiableExtrudedPolygon '{self.name}': "
                    f"partial_real_shape for axis {ax} is inferred from the "
                    f"initial vertex bounding box ({size:.3e} m). "
                    f"Do not specify it explicitly."
                )
            if grid_shape[ax] is not None:
                raise ValueError(
                    f"DifferentiableExtrudedPolygon '{self.name}': "
                    f"partial_grid_shape for axis {ax} is inferred from the "
                    f"initial vertex bounding box. Do not specify it explicitly."
                )
            real_shape[ax] = size
        object.__setattr__(self, "partial_real_shape", tuple(real_shape))
        object.__setattr__(self, "_n_vertices", int(verts_np.shape[0]))

    # ------------------------------------------------------------------ #
    # finalize(): call after placement, before any jit-compiled function  #
    # ------------------------------------------------------------------ #

    def finalize(self) -> "DifferentiableExtrudedPolygon":
        """Extract static grid geometry into plain numpy and return new instance.

        Must be called after placement constraints are solved (so that
        ``grid_slice_tuple`` and ``grid_shape`` are available) and before
        any ``jit``-compiled function that calls ``get_voxel_mask_for_shape``.

        Returns a new (frozen) instance with ``_h_centers_np``,
        ``_v_centers_np``, and ``_smoothing_hw`` populated.

        Example::

            objects = finalize_differentiable_polygons(objects)
        """
        h_centers = self._extract_centers_np(self.horizontal_axis)
        v_centers = self._extract_centers_np(self.vertical_axis)
        hw = self._extract_smoothing_hw()

        h_ax = self.horizontal_axis
        v_ax = self.vertical_axis
        real_shape = self.real_shape  # safe here — outside jit

        # aset returns a new frozen instance with the fields updated
        obj = self.aset("_h_centers_np", h_centers)
        obj = obj.aset("_v_centers_np", v_centers)
        obj = obj.aset("_smoothing_hw", hw)
        obj = obj.aset("_finalized", True)
        obj = obj.aset("_center_h", float(0.5 * real_shape[h_ax]))
        obj = obj.aset("_center_v", float(0.5 * real_shape[v_ax]))
        return obj

    def _extract_centers_np(self, ax: int) -> np.ndarray:
        """Extract grid cell centers as a plain numpy array.

        Called only from ``finalize()``, outside jit, after placement.
        ``grid.edges(ax)`` is called here on the real (non-traced) config.
        """
        lower, upper = self.grid_slice_tuple[ax]
        grid = self._config.resolved_grid
        if grid is None:
            spacing = float(self._config.uniform_spacing())
            n = self.grid_shape[ax]
            return (np.arange(n) + 0.5) * spacing
        # Call .edges() here in Python, outside jit — returns a real array
        edges = np.array(grid.edges(ax))  # force numpy, not jnp
        centers = 0.5 * (edges[lower:upper] + edges[lower + 1 : upper + 1])
        return centers - edges[lower]

    def _extract_smoothing_hw(self) -> float:
        """Extract smoothing half-width as a plain Python float."""
        if self.smoothing_width is not None:
            return float(self.smoothing_width) * 0.5
        grid = self._config.resolved_grid
        if grid is None:
            return 0.5 * float(self._config.uniform_spacing())
        h_lo, h_hi = self.grid_slice_tuple[self.horizontal_axis]
        v_lo, v_hi = self.grid_slice_tuple[self.vertical_axis]
        edges_h = np.array(grid.edges(self.horizontal_axis))
        edges_v = np.array(grid.edges(self.vertical_axis))
        spacing = 0.5 * (
            float(np.mean(np.diff(edges_h[h_lo : h_hi + 1]))) + float(np.mean(np.diff(edges_v[v_lo : v_hi + 1])))
        )
        return 0.5 * spacing

    # ------------------------------------------------------------------ #
    # Axis helpers                                                         #
    # ------------------------------------------------------------------ #

    @property
    def horizontal_axis(self) -> int:
        return get_transverse_axes(self.axis)[0]

    @property
    def vertical_axis(self) -> int:
        return get_transverse_axes(self.axis)[1]

    # ------------------------------------------------------------------ #
    # Differentiable SDF — safe under jit                                 #
    # ------------------------------------------------------------------ #

    def _polygon_sdf(
        self,
        px: jax.Array,  # (H,) — compile-time constant from numpy
        py: jax.Array,  # (V,) — compile-time constant from numpy
        verts: jax.Array,  # (N, 2) — the only traced quantity
    ) -> jax.Array:
        """Signed distance field, negative inside. Differentiable w.r.t. verts."""
        gx, gy = jnp.meshgrid(px, py, indexing="ij")
        points = jnp.stack([gx, gy], axis=-1)  # (H, V, 2)

        v0 = verts
        v1 = jnp.roll(verts, -1, axis=0)

        p = points[:, :, None, :]
        a = v0[None, None, :, :]
        b = v1[None, None, :, :]

        ab = b - a
        ap = p - a
        t = jnp.sum(ap * ab, axis=-1) / (jnp.sum(ab * ab, axis=-1) + 1e-30)
        t = jnp.clip(t, 0.0, 1.0)
        closest = a + t[..., None] * ab
        dist2 = jnp.sum((p - closest) ** 2, axis=-1)
        unsigned_dist = jnp.sqrt(jnp.min(dist2, axis=-1) + 1e-30)

        dx0 = v0[None, None, :, 0] - p[..., 0]
        dy0 = v0[None, None, :, 1] - p[..., 1]
        dx1 = v1[None, None, :, 0] - p[..., 0]
        dy1 = v1[None, None, :, 1] - p[..., 1]
        cross = dx0 * dy1 - dy0 * dx1
        dot = dx0 * dx1 + dy0 * dy1
        angle = jnp.arctan2(cross, dot + 1e-30)
        winding = jnp.sum(angle, axis=-1) / (2.0 * jnp.pi)

        sign = jnp.where(jnp.abs(winding) > 0.5, -1.0, 1.0)
        return sign * unsigned_dist

    # ------------------------------------------------------------------ #
    # StaticMultiMaterialObject interface                                  #
    # ------------------------------------------------------------------ #

    def get_voxel_mask_for_shape(self) -> jax.Array:
        h_ax = self.horizontal_axis
        v_ax = self.vertical_axis

        # Use pre-extracted numpy arrays if finalized (jit-safe path),
        # otherwise compute on the fly (placement path, outside jit).
        if self._finalized:
            assert self._h_centers_np is not None and self._v_centers_np is not None
            h_centers = jnp.asarray(self._h_centers_np)
            v_centers = jnp.asarray(self._v_centers_np)
            hw = self._smoothing_hw
        else:
            h_centers = jnp.asarray(self._extract_centers_np(h_ax))
            v_centers = jnp.asarray(self._extract_centers_np(v_ax))
            hw = self._extract_smoothing_hw()

        if self._finalized:
            center_h = self._center_h
            center_v = self._center_v
        else:
            center_h = float(0.5 * self.real_shape[h_ax])
            center_v = float(0.5 * self.real_shape[v_ax])
        grid_verts = self.vertices + jnp.array([center_h, center_v])

        sdf = self._polygon_sdf(h_centers, v_centers, grid_verts)
        fill_2d = 0.5 * (1.0 - jnp.tanh(sdf / (hw + 1e-30)))

        extrusion_height = self.grid_shape[self.axis]
        fill_2d_expanded = jnp.expand_dims(fill_2d, axis=self.axis)
        return jnp.repeat(fill_2d_expanded, repeats=extrusion_height, axis=self.axis)

    def get_material_mapping(self) -> jax.Array:
        """Uniform integer material index across all voxels."""
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        return jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
