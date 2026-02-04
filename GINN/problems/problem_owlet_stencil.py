import os
import torch
import numpy as np
import trimesh

from GINN.problems.constraints import (
    CompositeInterface, CompositeConstraint,
    SampleConstraint, SampleConstraintWithNormals
)
from GINN.problems.problem_base import ProblemBase
from util.misc import get_is_out_mask, get_default_device


class ProblemOwletStencil(ProblemBase):
    """
    Problem definition for Owlet acoustic stencil design.

    The stencil is a cylindrical cap with holes that create direction-specific
    acoustic signatures for DoA estimation.
    """

    def __init__(self,
                 nx,
                 n_points_envelope,
                 n_points_interfaces,
                 n_points_domain,
                 n_points_normals,
                 stencil_radius=0.013,  # 13mm
                 stencil_height=0.005,  # 5mm
                 min_hole_diameter=0.0005,  # 0.5mm
                 max_hole_diameter=0.002,  # 2mm
                 min_wall_thickness=0.0008,  # 0.8mm for 3D printing
                 nf_is_density=False,  # use SDF representation
                 **kwargs) -> None:
        super().__init__(nx=nx)

        self.n_points_envelope = n_points_envelope
        self.n_points_interfaces = n_points_interfaces
        self.n_points_domain = n_points_domain
        self.n_points_normals = n_points_normals

        # Stencil geometry parameters
        self.stencil_radius = stencil_radius
        self.stencil_height = stencil_height
        self.min_hole_diameter = min_hole_diameter
        self.max_hole_diameter = max_hole_diameter
        self.min_wall_thickness = min_wall_thickness

        self._envelope_constr = None
        self._interface_constraints = []
        self._normal_constraints = []
        self._obstacle_constraints = []
        device = get_default_device()

        # Define bounding box: cylinder centered at origin
        # x, y: circular cross-section, z: height
        margin = 0.002  # 2mm margin
        self.bounds = torch.tensor([
            [-stencil_radius - margin, stencil_radius + margin],  # x
            [-stencil_radius - margin, stencil_radius + margin],  # y
            [-margin, stencil_height + margin]  # z (base at z=0)
        ], dtype=torch.float32, device=device)

        # Store base and top centers
        self.base_center = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.top_center = torch.tensor([0.0, 0.0, stencil_height], device=device)

        # Generate constraint points
        # 1. Base interface (where stencil attaches to microphone)
        base_pts, base_normals = self._sample_base_interface(n_points_interfaces // 2, device)

        # 2. Top surface interface
        top_pts, top_normals = self._sample_top_interface(n_points_interfaces // 2, device)

        # 3. Side surface (cylindrical wall) - part of envelope
        side_pts, side_normals = self._sample_side_surface(n_points_envelope // 2, device)

        # 4. Envelope: outside the cylinder
        outside_pts = self._sample_outside_cylinder(n_points_envelope // 2, device)

        # 5. Inside envelope: volume where shape can exist (with holes)
        inside_pts = self._sample_inside_cylinder(n_points_domain, device)

        # 6. Domain: all space (for eikonal loss)
        domain_pts = torch.cat([inside_pts, outside_pts[:n_points_domain // 4]], dim=0)

        # Create constraints
        base_interface = SampleConstraintWithNormals(sample_pts=base_pts, normals=base_normals)
        top_interface = SampleConstraintWithNormals(sample_pts=top_pts, normals=top_normals)
        side_interface = SampleConstraintWithNormals(sample_pts=side_pts, normals=side_normals)

        all_interfaces = CompositeInterface([base_interface, top_interface])

        envelope_outside = SampleConstraint(sample_pts=outside_pts)
        inside_envelope = SampleConstraint(sample_pts=inside_pts)
        domain = SampleConstraint(sample_pts=domain_pts)

        # Store for visualization
        self.constr_pts_dict = {
            'base_interface': base_pts.cpu().numpy(),
            'top_interface': top_pts.cpu().numpy(),
            'side_surface': side_pts.cpu().numpy(),
            'outside_envelope': outside_pts.cpu().numpy(),
            'inside_envelope': inside_pts.cpu().numpy(),
            'interface': torch.cat([base_pts, top_pts], dim=0).cpu().numpy(),
            'domain': domain_pts.cpu().numpy()
        }

        # Save constraints
        self._envelope_constr = [envelope_outside]
        self._interface_constraints = [base_interface, top_interface]
        self._obstacle_constraints = None
        self._inside_envelope = inside_envelope
        self._domain = domain

    def _sample_base_interface(self, n_points, device):
        """Sample points on the base circle (microphone interface)"""
        # Circle in xy-plane at z=0
        angles = torch.linspace(0, 2 * np.pi, n_points, device=device)
        x = self.stencil_radius * torch.cos(angles)
        y = self.stencil_radius * torch.sin(angles)
        z = torch.zeros_like(x)

        pts = torch.stack([x, y, z], dim=1)

        # Normals point downward (into microphone)
        normals = torch.zeros_like(pts)
        normals[:, 2] = -1.0

        return pts, normals

    def _sample_top_interface(self, n_points, device):
        """Sample points on top surface of the stencil"""
        # Circle in xy-plane at z=stencil_height
        angles = torch.linspace(0, 2 * np.pi, n_points, device=device)
        x = self.stencil_radius * torch.cos(angles)
        y = self.stencil_radius * torch.sin(angles)
        z = torch.ones_like(x) * self.stencil_height

        pts = torch.stack([x, y, z], dim=1)

        # Normals point upward
        normals = torch.zeros_like(pts)
        normals[:, 2] = 1.0

        return pts, normals

    def _sample_side_surface(self, n_points, device):
        """Sample points on cylindrical side surface"""
        # Cylindrical surface
        angles = torch.rand(n_points, device=device) * 2 * np.pi
        heights = torch.rand(n_points, device=device) * self.stencil_height

        x = self.stencil_radius * torch.cos(angles)
        y = self.stencil_radius * torch.sin(angles)
        z = heights

        pts = torch.stack([x, y, z], dim=1)

        # Normals point outward radially
        normals = torch.stack([torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=1)

        return pts, normals

    def _sample_outside_cylinder(self, n_points, device):
        """Sample points outside the cylindrical envelope"""
        pts = torch.rand(n_points, 3, device=device)

        # Scale to bounding box
        for i in range(3):
            pts[:, i] = pts[:, i] * (self.bounds[i, 1] - self.bounds[i, 0]) + self.bounds[i, 0]

        # Keep only points outside cylinder
        dist_from_axis = torch.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        outside_mask = (dist_from_axis > self.stencil_radius) | (pts[:, 2] < 0) | (pts[:, 2] > self.stencil_height)

        outside_pts = pts[outside_mask]

        # If not enough points, generate more
        while len(outside_pts) < n_points:
            additional = torch.rand(n_points, 3, device=device)
            for i in range(3):
                additional[:, i] = additional[:, i] * (self.bounds[i, 1] - self.bounds[i, 0]) + self.bounds[i, 0]

            dist_from_axis = torch.sqrt(additional[:, 0]**2 + additional[:, 1]**2)
            outside_mask = (dist_from_axis > self.stencil_radius) | (additional[:, 2] < 0) | (additional[:, 2] > self.stencil_height)
            outside_pts = torch.cat([outside_pts, additional[outside_mask]], dim=0)

        return outside_pts[:n_points]

    def _sample_inside_cylinder(self, n_points, device):
        """Sample points inside the cylindrical volume"""
        # Sample in cylinder
        pts = []

        while len(pts) < n_points:
            # Sample in bounding box
            candidates = torch.rand(n_points * 2, 3, device=device)

            # Scale to cylinder dimensions
            angles = candidates[:, 0] * 2 * np.pi
            radii = torch.sqrt(candidates[:, 1]) * self.stencil_radius  # sqrt for uniform distribution
            heights = candidates[:, 2] * self.stencil_height

            x = radii * torch.cos(angles)
            y = radii * torch.sin(angles)
            z = heights

            cylinder_pts = torch.stack([x, y, z], dim=1)
            pts.append(cylinder_pts)

        pts = torch.cat(pts, dim=0)
        return pts[:n_points]

    def is_inside_envelope(self, pts):
        """Check if points are inside the cylindrical envelope"""
        dist_from_axis = torch.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        inside_mask = (dist_from_axis <= self.stencil_radius) & (pts[:, 2] >= 0) & (pts[:, 2] <= self.stencil_height)
        return inside_mask
