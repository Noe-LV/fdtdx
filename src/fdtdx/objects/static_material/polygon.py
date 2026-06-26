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

    The bounding box is fixed from the *initial* vertices and does not update
    as vertices move during optimization. The allocated grid region therefore
    stays constant; vertices should remain within it.

    Args:
        material_name: Name of the material in the ``materials`` dict.
        axis: Extrusion axis (0, 1, or 2).
        vertices: ``(N, 2)`` array of polygon vertices in metrical units
            (meters), centered at the origin. Non-frozen — participates in
            JAX differentiation.
        smoothing_width: Width of the SDF transition band in meters.
            ``None`` (default) uses one grid cell width, computed from the
            config when the mask is first requested.
    """

    # ------------------------------------------------------------------ #
    # Public constructor fields                                            #
    # ------------------------------------------------------------------ #

    #: Material name in the materials dictionary.
    material_name: str = frozen_field()

    #: Extrusion axis (0=x, 1=y, 2=z).
    axis: int = frozen_field()

    #: (N, 2) polygon vertices in meters, centered at origin.
    #: Non-frozen so JAX can differentiate through it.
    vertices: jax.Array = field()

    #: Optional explicit SDF transition width (meters).
    #: None → one grid cell, computed from config at mask-generation time.
    smoothing_width: float | None = frozen_field(default=None)

    # ------------------------------------------------------------------ #
    # Private fields frozen at __post_init__                              #
    # ------------------------------------------------------------------ #

    #: Number of polygon vertices. Static Python int, frozen at init.
    #: Never read from verts.shape[0] inside jit.
    _n_vertices: int = frozen_field(default=0, init=False)

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        verts_np = np.asarray(self.vertices)

        # ---- bounding box → allocate grid region ---- #
        # grid_slice_tuple is NOT available yet (object not placed),
        # but real_shape only needs the vertex bounding box.
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

        # ---- static vertex count ---- #
        # Frozen here so _polygon_sdf never reads verts.shape[0] under jit.
        object.__setattr__(self, "_n_vertices", int(verts_np.shape[0]))

    # ------------------------------------------------------------------ #
    # Static axis helpers                                                  #
    # ------------------------------------------------------------------ #

    @property
    def horizontal_axis(self) -> int:
        """Horizontal axis perpendicular to the extrusion axis."""
        return get_transverse_axes(self.axis)[0]

    @property
    def vertical_axis(self) -> int:
        """Vertical axis perpendicular to the extrusion axis."""
        return get_transverse_axes(self.axis)[1]

    # ------------------------------------------------------------------ #
    # Grid center computation — numpy, called after object is placed      #
    # ------------------------------------------------------------------ #

    def _compute_grid_centers_np(self, ax: int) -> np.ndarray:
        """Grid cell centers relative to this object's lower edge.

        Uses only static config/placement values (plain numpy). Safe to call
        any time after the object has been placed in the grid (i.e. after
        constraints are solved). Never called inside jit — result is wrapped
        in jnp.asarray() at the call site, which JAX treats as a
        compile-time constant since the numpy array itself is not a traced
        leaf.
        """
        lower, upper = self.grid_slice_tuple[ax]  # available post-placement
        grid = self._config.resolved_grid
        if grid is None:
            spacing = float(self._config.uniform_spacing())
            n = self.grid_shape[ax]
            return (np.arange(n) + 0.5) * spacing
        edges = np.asarray(grid.edges(ax))
        centers = 0.5 * (edges[lower:upper] + edges[lower + 1 : upper + 1])
        return centers - edges[lower]

    def _compute_smoothing_hw(self) -> float:
        """Smoothing half-width in meters, derived from config post-placement."""
        if self.smoothing_width is not None:
            return float(self.smoothing_width) * 0.5
        grid = self._config.resolved_grid
        if grid is None:
            spacing = float(self._config.uniform_spacing())
        else:
            h_lo, h_hi = self.grid_slice_tuple[self.horizontal_axis]
            v_lo, v_hi = self.grid_slice_tuple[self.vertical_axis]
            edges_h = np.asarray(grid.edges(self.horizontal_axis))
            edges_v = np.asarray(grid.edges(self.vertical_axis))
            spacing = 0.5 * (
                float(np.mean(np.diff(edges_h[h_lo : h_hi + 1]))) + float(np.mean(np.diff(edges_v[v_lo : v_hi + 1])))
            )
        return 0.5 * spacing

    # ------------------------------------------------------------------ #
    # Differentiable polygon SDF                                          #
    # ------------------------------------------------------------------ #

    def _polygon_sdf(
        self,
        px: jax.Array,  # (H,) horizontal grid centers — compile-time constant
        py: jax.Array,  # (V,) vertical grid centers — compile-time constant
        verts: jax.Array,  # (N, 2) — the only traced quantity
    ) -> jax.Array:
        """Signed distance field for a 2-D polygon evaluated on a grid.

        Returns ``(H, V)``, negative inside the polygon. Fully differentiable
        w.r.t. ``verts``. The vertex count comes from the frozen
        ``_n_vertices`` field — never from ``verts.shape[0]`` — so this is
        safe under ``jit``.
        """
        gx, gy = jnp.meshgrid(px, py, indexing="ij")
        points = jnp.stack([gx, gy], axis=-1)  # (H, V, 2)

        v0 = verts  # (N, 2) edge starts
        v1 = jnp.roll(verts, -1, axis=0)  # (N, 2) edge ends

        p = points[:, :, None, :]  # (H, V, 1, 2)
        a = v0[None, None, :, :]  # (H, V, N, 2)
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
        """Soft voxel fill-fraction mask, differentiable w.r.t. ``vertices``.

        Returns float ``[0, 1]`` of shape ``grid_shape``.

        Grid centers and smoothing width are computed here from static numpy
        config values and wrapped in ``jnp.asarray()``. JAX treats these as
        compile-time constants (they are not traced leaves), so the only
        quantity that actually flows through the JAX trace is ``self.vertices``.
        """
        h_ax = self.horizontal_axis
        v_ax = self.vertical_axis

        # Compile-time constants: numpy → jnp.asarray is a no-op in the trace
        h_centers = jnp.asarray(self._compute_grid_centers_np(h_ax))
        v_centers = jnp.asarray(self._compute_grid_centers_np(v_ax))
        hw = self._compute_smoothing_hw()  # plain Python float

        # Only self.vertices is traced beyond this point
        center_h = 0.5 * self.real_shape[h_ax]  # static float
        center_v = 0.5 * self.real_shape[v_ax]  # static float
        grid_verts = self.vertices + jnp.array([center_h, center_v])

        sdf = self._polygon_sdf(h_centers, v_centers, grid_verts)

        fill_2d = 0.5 * (1.0 - jnp.tanh(sdf / (hw + 1e-30)))

        extrusion_height = self.grid_shape[self.axis]  # static int
        fill_2d_expanded = jnp.expand_dims(fill_2d, axis=self.axis)
        return jnp.repeat(fill_2d_expanded, repeats=extrusion_height, axis=self.axis)

    def get_material_mapping(self) -> jax.Array:
        """Uniform integer material index across all voxels."""
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        return jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
