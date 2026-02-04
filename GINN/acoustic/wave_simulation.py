"""
Simplified acoustic wave simulation for Owlet stencil design.

This module provides differentiable approximations of acoustic wave propagation
through the stencil geometry to compute frequency-dependent directional responses.
"""

import torch
import numpy as np
from typing import Tuple, List


class AcousticSimulator:
    """
    Simplified acoustic simulator for stencil frequency response.

    Uses analytical approximations for:
    1. Diffraction through small apertures (holes)
    2. Phase delays from different path lengths
    3. Capillary/resonance effects from hole geometry
    """

    def __init__(self,
                 frequencies: List[float],
                 angles: List[float],
                 speed_of_sound: float = 343.0,  # m/s in air
                 stencil_radius: float = 0.013,
                 stencil_height: float = 0.005):
        """
        Args:
            frequencies: List of frequencies to evaluate (Hz)
            angles: List of angles to evaluate (degrees, 0=front)
            speed_of_sound: Speed of sound in medium (m/s)
            stencil_radius: Radius of stencil cap (m)
            stencil_height: Height of stencil cap (m)
        """
        self.frequencies = torch.tensor(frequencies, dtype=torch.float32)
        self.angles_deg = torch.tensor(angles, dtype=torch.float32)
        self.angles_rad = self.angles_deg * np.pi / 180.0
        self.c = speed_of_sound
        self.stencil_radius = stencil_radius
        self.stencil_height = stencil_height

        # Precompute wavelengths
        self.wavelengths = self.c / self.frequencies  # λ = c/f

    def compute_frequency_response(self,
                                    surface_points: torch.Tensor,
                                    surface_normals: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency response for all angles and frequencies.

        This is a simplified model that:
        1. Identifies hole-like features from the surface geometry
        2. Computes diffraction and phase delays
        3. Sums contributions from all holes

        Args:
            surface_points: (N, 3) tensor of surface points
            surface_normals: (N, 3) tensor of surface normals

        Returns:
            response: (n_angles, n_frequencies) complex tensor
                     representing frequency gains for each direction
        """
        device = surface_points.device

        # Move precomputed tensors to device
        frequencies = self.frequencies.to(device)
        angles_rad = self.angles_rad.to(device)
        wavelengths = self.wavelengths.to(device)

        n_angles = len(angles_rad)
        n_freqs = len(frequencies)

        # Detect holes by finding regions where surface is nearly horizontal
        # (top surface with holes will have varying normals)
        top_surface_mask = (surface_points[:, 2] > self.stencil_height * 0.7) & \
                           (surface_points[:, 2] < self.stencil_height * 1.3)

        if top_surface_mask.sum() == 0:
            # No top surface detected, return default response
            return torch.ones(n_angles, n_freqs, dtype=torch.complex64, device=device)

        top_points = surface_points[top_surface_mask]
        top_normals = surface_normals[top_surface_mask]

        # Cluster points to identify individual holes
        hole_centers = self._identify_hole_centers(top_points, top_normals)

        if len(hole_centers) == 0:
            # No holes, return uniform response
            return torch.ones(n_angles, n_freqs, dtype=torch.complex64, device=device)

        # Compute response for each angle and frequency
        response = torch.zeros(n_angles, n_freqs, dtype=torch.complex64, device=device)

        for i, angle in enumerate(angles_rad):
            for j, (freq, wavelength) in enumerate(zip(frequencies, wavelengths)):
                # Incident wave direction (in xz plane for simplicity)
                k_incident = 2 * np.pi / wavelength
                wave_dir = torch.tensor([torch.sin(angle), 0.0, torch.cos(angle)], device=device)

                # Sum contributions from all holes
                complex_sum = torch.tensor(0.0 + 0.0j, dtype=torch.complex64, device=device)

                for hole_center in hole_centers:
                    # Path length difference
                    path_diff = torch.dot(hole_center, wave_dir)

                    # Phase delay
                    phase = k_incident * path_diff

                    # Diffraction amplitude (simplified - assumes circular apertures)
                    # For small holes: amplitude ∝ (hole_area / wavelength^2)
                    hole_radius = self._estimate_hole_radius(hole_center, top_points)
                    diffraction_amp = (np.pi * hole_radius**2) / (wavelength**2 + 1e-6)

                    # Directional pattern (simple cosine for now)
                    # Real diffraction has more complex patterns
                    directivity = torch.abs(torch.cos(angle)) + 0.3  # bias towards frontal

                    # Complex contribution
                    amplitude = diffraction_amp * directivity
                    complex_sum += amplitude * torch.exp(1j * phase)

                response[i, j] = complex_sum

        # Normalize
        response = response / (torch.abs(response).max() + 1e-6)

        return response

    def _identify_hole_centers(self,
                                top_points: torch.Tensor,
                                top_normals: torch.Tensor,
                                cluster_threshold: float = 0.003) -> List[torch.Tensor]:
        """
        Identify hole centers by clustering top surface points.

        Args:
            top_points: Points on top surface
            top_normals: Normals at top surface
            cluster_threshold: Distance threshold for clustering (m)

        Returns:
            List of hole center positions (3D tensors)
        """
        if len(top_points) == 0:
            return []

        # Simple clustering: grid-based binning
        device = top_points.device

        # Project to 2D (xy plane)
        xy_points = top_points[:, :2]

        # Create grid
        n_bins = 20
        x_min, x_max = -self.stencil_radius, self.stencil_radius
        y_min, y_max = -self.stencil_radius, self.stencil_radius

        x_bins = torch.linspace(x_min, x_max, n_bins, device=device)
        y_bins = torch.linspace(y_min, y_max, n_bins, device=device)

        # Assign points to bins
        x_indices = torch.searchsorted(x_bins, xy_points[:, 0])
        y_indices = torch.searchsorted(y_bins, xy_points[:, 1])

        # Find non-empty bins
        hole_centers = []
        visited = set()

        for i in range(len(top_points)):
            bin_id = (x_indices[i].item(), y_indices[i].item())

            if bin_id not in visited:
                visited.add(bin_id)

                # Find all points in this bin and nearby bins
                mask = (torch.abs(x_indices - x_indices[i]) <= 1) & \
                       (torch.abs(y_indices - y_indices[i]) <= 1)

                if mask.sum() > 3:  # Require at least a few points
                    cluster_points = top_points[mask]
                    center = cluster_points.mean(dim=0)
                    hole_centers.append(center)

        return hole_centers

    def _estimate_hole_radius(self,
                               hole_center: torch.Tensor,
                               top_points: torch.Tensor,
                               max_radius: float = 0.003) -> float:
        """
        Estimate hole radius from nearby surface points.

        Args:
            hole_center: Center of hole (3D)
            top_points: All top surface points
            max_radius: Maximum expected hole radius (m)

        Returns:
            Estimated radius (scalar)
        """
        # Find points near this hole
        distances = torch.norm(top_points[:, :2] - hole_center[:2], dim=1)
        nearby_mask = distances < max_radius

        if nearby_mask.sum() == 0:
            return 0.001  # Default 1mm radius

        # Estimate radius as mean distance of nearby points
        nearby_distances = distances[nearby_mask]
        estimated_radius = nearby_distances.mean().item()

        return max(0.0005, min(estimated_radius, 0.002))  # Clamp to [0.5mm, 2mm]

    def compute_angular_diversity(self, response: torch.Tensor) -> torch.Tensor:
        """
        Compute angular diversity metric (how distinguishable are different angles).

        Args:
            response: (n_angles, n_frequencies) complex tensor

        Returns:
            diversity: scalar tensor (higher = more diverse)
        """
        n_angles = response.shape[0]

        # Compute pairwise distances between angle responses
        diversity_sum = 0.0

        for i in range(n_angles):
            for j in range(i + 1, n_angles):
                # Euclidean distance in frequency space
                diff = torch.abs(response[i] - response[j])
                diversity_sum += diff.mean()

        # Normalize by number of pairs
        n_pairs = n_angles * (n_angles - 1) / 2
        diversity = diversity_sum / (n_pairs + 1e-6)

        return diversity

    def compute_directional_selectivity(self,
                                         response: torch.Tensor,
                                         target_angle_idx: int = 0) -> torch.Tensor:
        """
        Compute how well the response distinguishes target angle from others.

        Args:
            response: (n_angles, n_frequencies) complex tensor
            target_angle_idx: Index of target angle (default 0 = front)

        Returns:
            selectivity: scalar tensor (higher = better selectivity)
        """
        target_response = response[target_angle_idx]

        # Compute average response of off-target angles
        off_target_indices = [i for i in range(response.shape[0]) if i != target_angle_idx]
        off_target_response = response[off_target_indices].mean(dim=0)

        # Selectivity = difference between target and off-target
        selectivity = torch.abs(target_response - off_target_response).mean()

        return selectivity
