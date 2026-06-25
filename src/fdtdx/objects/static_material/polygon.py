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
    """A differentiable extruded polygon for shape optimization.

    Unlike ExtrudedPolygon, the vertices are a JAX array and are not frozen,
    so gradients can be computed with respect to vertex positions. The mask
    returned by get_voxel_mask_for_shape is a soft float mask (fill fractions
    in [0, 1]) computed via a smooth polygon SDF, enabling gradient-based
    shape optimization.

    The bounding box is fixed at construction from the initial vertices and
    does not update as vertices move. Vertices should stay within the
    originally allocated grid region during optimization.

    Args:
        material_name: Name of the material in the materials dictionary.
        axis: The extrusion axis (0, 1, or 2).
        vertices: JAX array of shape (N, 2) with vertices in metrical units
            (meters), centered at the origin. This is a regular (non-frozen)
            field so it participates in JAX differentiation.
        smoothing_width: Width of the SDF-to-fill-fraction transition in
            meters. Defaults to None, which uses one grid cell width.
    """

    material_name: str = frozen_field()
    axis: int = frozen_field()

    # Regular field — not frozen — so JAX can differentiate through it.
    vertices: jax.Array = field()

    # Optional explicit smoothing width (meters). None → one grid cell.
    smoothing_width: float | None = frozen_field(default=None)

    def __post_init__(self):
        # Bounding box is computed once from the *initial* vertices and locked
        # in. During optimization vertices may shift, but the allocated grid
        # region stays fixed.
        verts = np.asarray(self.vertices)  # concrete at init time
        w = float(verts[:, 0].max() - verts[:, 0].min())
        h = float(verts[:, 1].max() - verts[:, 1].min())
        real_shape = list(self.partial_real_shape)
        grid_shape = list(self.partial_grid_shape)
        for ax, size in ((self.horizontal_axis, w), (self.vertical_axis, h)):
            if real_shape[ax] is not None:
                raise ValueError(
                    f"DifferentiableExtrudedPolygon {self.name}: "
                    f"partial_real_shape for axis {ax} is inferred from the "
                    f"initial vertex bounding box ({size:.3e} m). "
                    f"Do not specify it explicitly."
                )
            if grid_shape[ax] is not None:
                raise ValueError(
                    f"DifferentiableExtrudedPolygon {self.name}: "
                    f"partial_grid_shape for axis {ax} is inferred from the "
                    f"initial vertex bounding box. Do not specify it explicitly."
                )
            real_shape[ax] = size
        object.__setattr__(self, "partial_real_shape", tuple(real_shape))

    @property
    def horizontal_axis(self) -> int:
        return get_transverse_axes(self.axis)[0]

    @property
    def vertical_axis(self) -> int:
        return get_transverse_axes(self.axis)[1]

    def _grid_centers(self, ax: int) -> jax.Array:
        """Physical cell centers relative to this object's lower edge, for axis ax."""
        lower, upper = self.grid_slice_tuple[ax]
        grid = self._config.resolved_grid
        if grid is None:
            spacing = self._config.uniform_spacing()
            return (jnp.arange(self.grid_shape[ax]) + 0.5) * spacing
        edges = jnp.asarray(grid.edges(ax))
        return 0.5 * (edges[lower:upper] + edges[lower + 1 : upper + 1]) - edges[lower]

    def _polygon_sdf(
        self,
        px: jax.Array,  # (H,) horizontal coords of query points
        py: jax.Array,  # (V,) vertical coords of query points
        verts: jax.Array,  # (N, 2) polygon vertices
    ) -> jax.Array:
        """Signed distance field for a 2-D polygon, evaluated on a grid.

        Positive outside, negative inside (sign convention matches the
        standard "point-in-polygon via winding" approach). Returns array
        of shape (H, V).

        The SDF is computed as:
          - unsigned distance  = min over edges of distance to segment
          - sign               = determined by winding number (inside → negative)
        Both parts are differentiable through JAX w.r.t. verts.
        """
        # Build grid: shape (H, V, 2)
        gx, gy = jnp.meshgrid(px, py, indexing="ij")
        points = jnp.stack([gx, gy], axis=-1)  # (H, V, 2)

        # Edge start/end: both (N, 2)
        v0 = verts  # edge starts
        v1 = jnp.roll(verts, -1, axis=0)  # edge ends

        # Expand for broadcasting: points (H, V, 1, 2), edges (1, 1, N, 2)
        p = points[:, :, None, :]  # (H, V, 1, 2)
        a = v0[None, None, :, :]  # (H, V, N, 2)
        b = v1[None, None, :, :]  # (H, V, N, 2)

        ab = b - a  # (H, V, N, 2)  edge vector
        ap = p - a  # (H, V, N, 2)  point-to-start

        # Scalar projection t ∈ [0, 1] along each edge
        t = jnp.sum(ap * ab, axis=-1) / (jnp.sum(ab * ab, axis=-1) + 1e-30)
        t = jnp.clip(t, 0.0, 1.0)  # (H, V, N)

        # Closest point on edge segment
        closest = a + t[..., None] * ab  # (H, V, N, 2)

        # Squared distance to closest point on each edge
        dist2 = jnp.sum((p - closest) ** 2, axis=-1)  # (H, V, N)
        unsigned_dist = jnp.sqrt(jnp.min(dist2, axis=-1) + 1e-30)  # (H, V)

        # Winding number for sign: sum cross-products over edges.
        # A point is inside when winding_number != 0.
        # Using the standard ray-crossing approach in a differentiable way
        # via the angle-summation form.
        dx0 = v0[None, None, :, 0] - p[..., 0]  # (H, V, N)
        dy0 = v0[None, None, :, 1] - p[..., 1]
        dx1 = v1[None, None, :, 0] - p[..., 0]
        dy1 = v1[None, None, :, 1] - p[..., 1]
        cross = dx0 * dy1 - dy0 * dx1  # (H, V, N)
        dot = dx0 * dx1 + dy0 * dy1  # (H, V, N)
        angle = jnp.arctan2(cross, dot + 1e-30)  # (H, V, N)
        winding = jnp.sum(angle, axis=-1) / (2.0 * jnp.pi)  # (H, V)

        # Inside → winding ≠ 0 → sign = -1
        sign = jnp.where(jnp.abs(winding) > 0.5, -1.0, 1.0)
        return sign * unsigned_dist  # (H, V), negative inside

    def get_voxel_mask_for_shape(self) -> jax.Array:
        """Soft voxel mask via polygon SDF.

        Returns a float array in [0, 1] of shape ``grid_shape``.  Interior
        voxels → 1, exterior → 0, boundary voxels → smooth fill fraction.
        The result is differentiable with respect to ``self.vertices``.
        """
        h_ax = self.horizontal_axis
        v_ax = self.vertical_axis

        h_centers = self._grid_centers(h_ax)  # (H,)
        v_centers = self._grid_centers(v_ax)  # (V,)

        # Shift vertices from object-center coords to local grid coords
        # (same shift as the original ExtrudedPolygon).
        center_h = 0.5 * self.real_shape[h_ax]
        center_v = 0.5 * self.real_shape[v_ax]
        grid_verts = self.vertices + jnp.array([center_h, center_v])

        # Signed distance field: (H, V), negative inside polygon
        sdf = self._polygon_sdf(h_centers, v_centers, grid_verts)

        # Smoothing half-width: default = one grid cell
        if self.smoothing_width is not None:
            hw = self.smoothing_width * 0.5
        else:
            grid = self._config.resolved_grid
            if grid is None:
                spacing = self._config.uniform_spacing()
            else:
                # Use mean spacing in the two transverse directions as proxy
                h_lo, h_hi = self.grid_slice_tuple[h_ax]
                v_lo, v_hi = self.grid_slice_tuple[v_ax]
                h_edges = jnp.asarray(grid.edges(h_ax))
                v_edges = jnp.asarray(grid.edges(v_ax))
                spacing = 0.5 * (
                    jnp.mean(jnp.diff(h_edges[h_lo : h_hi + 1])) + jnp.mean(jnp.diff(v_edges[v_lo : v_hi + 1]))
                )
            hw = 0.5 * spacing

        # Smooth step: fill fraction goes from 1 (deep inside) to 0 (outside)
        # using a cosine ramp over the transition band [-hw, +hw].
        # sdf < 0 → inside; sdf > 0 → outside.
        fill_2d = 0.5 * (1.0 - jnp.tanh(sdf / (hw + 1e-30)))  # (H, V)

        # Extrude along the fiber axis
        extrusion_height = self.grid_shape[self.axis]
        fill_2d_expanded = jnp.expand_dims(fill_2d, axis=self.axis)  # (H, 1, V) or similar
        mask = jnp.repeat(fill_2d_expanded, repeats=extrusion_height, axis=self.axis)

        return mask  # float32 in [0, 1], shape == grid_shape

    def get_material_mapping(self) -> jax.Array:
        all_names = compute_ordered_names(self.materials)
        idx = all_names.index(self.material_name)
        return jnp.ones(self.grid_shape, dtype=jnp.int32) * idx
