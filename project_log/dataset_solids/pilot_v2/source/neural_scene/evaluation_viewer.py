"""Named comparison layers on the installed DynaViewer's window/recording."""

import numpy as np

from .evaluation_mesh import mesh_in_robot


COLORS = {"prediction": (56 / 255, 217 / 255, 239 / 255),
          "observation": (120 / 255, 239 / 255, 128 / 255),
          "target": (1., 171 / 255, 69 / 255), "mesh": (.65, .68, .73)}
NAMES = ("prediction", "observation", "target", "mesh")


class ComparisonViewer:
    """Keep DynaViewer's camera/lifecycle while drawing semantic layers."""

    def __init__(self, renderer, layers, opacities, **options):
        from viewer.history import create_history_viewer

        self.renderer = renderer
        self.layers = tuple(layers)
        self.opacities = dict(opacities)
        self._restored = dict(opacities)
        self._selected = self.layers[0]
        self._mesh = None
        self._visuals = {}
        self._step = 0
        self.window = create_history_viewer(renderer=renderer, **options)
        if renderer == "vispy":
            from vispy import scene
            try:
                for name in self.layers:
                    points = scene.visuals.Markers(parent=self.window._scene)
                    points.set_gl_state("translucent", depth_test=True, depth_mask=False)
                    points.order = 20
                    points.visible = False
                    self._visuals[name] = (points,) + self.window._make_voxel_visual()
                self._mesh_visual = scene.visuals.Mesh(parent=self.window._scene, shading="flat")
                self._mesh_visual.set_gl_state("translucent", depth_test=True, depth_mask=False)
                self._mesh_visual.order = 0
                self._mesh_visual.visible = False
                self._visuals["mesh"] = (self._mesh_visual,)
                self._hud = scene.visuals.Text(parent=self.window.canvas.scene, color="white",
                    font_size=10, anchor_x="left", anchor_y="top", pos=(12, 12))
                self.window.canvas.events.key_press.connect(self._key_press)
                for name in self._visuals:
                    self.set_opacity(name, self.opacities[name])
            except BaseException:
                self.window.close()
                raise

    def __getattr__(self, name):
        return getattr(self.window, name)

    def set_mesh(self, mesh):
        """Replace world-space geometry between scenes; None clears the mesh."""
        self._mesh = mesh
        if self.renderer == "vispy":
            self._mesh_visual.visible = False
            self._refresh_hud()

    def set_opacity(self, name, value):
        if name not in NAMES or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Opacity must be finite and between 0 and 1")
        self.opacities[name] = float(value)
        for visual in self._visuals.get(name, ()):
            visual.opacity = float(value)
        if self.renderer == "vispy":
            self._refresh_hud()
            self.window.canvas.update()

    def _refresh_hud(self):
        active = self.layers + (("mesh",) if self._mesh is not None else ())
        rows = ["1 prediction | 2 observation | 3 GT | 4 mesh",
                "+/- opacity (10%) | 0 hide/restore | R recenter | Esc close"]
        for name in active:
            rows.append("{} {}: {:.0%}".format(
                ">" if name == self._selected else " ", "GT" if name == "target" else name,
                self.opacities[name]))
        self._hud.text = "\n".join(rows)

    def _key_press(self, event):
        key = event.key.name if event.key is not None else ""
        char = getattr(event, "text", "")
        if key in ("1", "2", "3", "4"):
            name = NAMES[int(key) - 1]
            if name in self.layers or (name == "mesh" and self._mesh is not None):
                self._selected = name
                self._refresh_hud()
        elif char in ("+", "=", "-") or key in ("+", "=", "-", "Up", "Down"):
            delta = -.1 if char == "-" or key in ("-", "Down") else .1
            self.set_opacity(self._selected, np.clip(self.opacities[self._selected] + delta, 0, 1))
        elif key == "0":
            value = self.opacities[self._selected]
            if value:
                self._restored[self._selected] = value
            self.set_opacity(self._selected, 0 if value else self._restored[self._selected] or 1.)

    def update(self, snapshot):
        if self.renderer == "vispy":
            self._update_vispy(snapshot)
        else:
            self._update_rerun(snapshot)

    def _update_vispy(self, snapshot):
        for name, layer in snapshot.layers.items():
            points, faces, edges = self._visuals[name]
            voxels = getattr(layer, "voxel_centers", None)
            if voxels is None:
                faces.visible = edges.visible = False
                self.window._set_points(points, layer.points, COLORS[name])
            else:
                points.visible = False
                # Faces retain their translucent fill; opacity also scales edges.
                self.window._set_voxels((faces, edges), voxels, layer, layer.sensor_pose,
                                        COLORS[name], .25)
        if self._mesh is not None:
            vertices, faces = self._mesh
            self._mesh_visual.set_data(vertices=mesh_in_robot(vertices, snapshot.current.sensor_pose),
                                       faces=faces, color=COLORS["mesh"] + (1.,))
            self._mesh_visual.visible = True
        self.window.canvas.update()
        self.window.app.process_events()

    def _update_rerun(self, snapshot):
        rr, recording = self.window.rr, self.window.recording
        # A monotonic index also distinguishes scenes/loops with repeated frame IDs.
        recording.set_time_sequence("playback", self._step)
        self._step += 1
        recording.set_time_sequence("frame", int(snapshot.current.frame_id))
        recording.set_time_nanos("simulation", int(snapshot.current.timestamp_ns))
        for name, layer in snapshot.layers.items():
            path = "current_sensor/" + name
            recording.log(path, rr.Clear(recursive=True))
            alpha = self.opacities[name]
            voxels = getattr(layer, "voxel_centers", None)
            centers = layer.points if voxels is None else voxels
            if not len(centers) or alpha == 0:
                continue
            stride = max(1, (len(centers) + self.window.max_points - 1) // self.window.max_points)
            centers = centers[::stride]
            color = [round(c * 255) for c in COLORS[name]]
            if voxels is None:
                recording.log(path, rr.Points3D(centers, colors=color + [round(alpha * 255)],
                    radii=rr.Radius.ui_points(self.window.point_size / 2)))
            else:
                for suffix, opacity, fill in (("faces", .25 * alpha, rr.components.FillMode.Solid),
                                              ("edges", alpha, rr.components.FillMode.MajorWireframe)):
                    recording.log(path + "/" + suffix, rr.Boxes3D(centers=centers,
                        half_sizes=np.full((len(centers), 3), layer.voxel_size_m / 2, dtype=np.float32),
                        colors=color + [round(opacity * 255)], fill_mode=fill))
        recording.log("current_sensor/mesh", rr.Clear(recursive=True))
        if self._mesh is not None and self.opacities["mesh"]:
            vertices, faces = self._mesh
            recording.log("current_sensor/mesh", rr.Mesh3D(
                vertex_positions=mesh_in_robot(vertices, snapshot.current.sensor_pose),
                triangle_indices=faces, albedo_factor=COLORS["mesh"] + (self.opacities["mesh"],)))
