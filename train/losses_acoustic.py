"""
Acoustic loss functions for Owlet stencil optimization.

These losses guide the neural field to generate stencil geometries
with desired acoustic frequency response properties.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from GINN.acoustic.wave_simulation import AcousticSimulator


class AcousticLosses:
    """
    Collection of acoustic-specific loss functions for stencil design.
    """

    def __init__(self,
                 frequencies: list = None,
                 angles: list = None,
                 target_response: torch.Tensor = None,
                 stencil_radius: float = 0.013,
                 stencil_height: float = 0.005):
        """
        Args:
            frequencies: List of frequencies to evaluate (Hz)
            angles: List of angles to evaluate (degrees)
            target_response: Optional target frequency response (n_angles, n_frequencies)
            stencil_radius: Radius of stencil (m)
            stencil_height: Height of stencil (m)
        """
        if frequencies is None:
            frequencies = [500, 1000, 2000, 4000, 8000]  # Audible range
        if angles is None:
            angles = [0, 30, 60, 90, 120, 150, 180]  # 7 angles

        self.simulator = AcousticSimulator(
            frequencies=frequencies,
            angles=angles,
            stencil_radius=stencil_radius,
            stencil_height=stencil_height
        )

        self.target_response = target_response
        self.frequencies = frequencies
        self.angles = angles

    def compute_frequency_response_loss(self,
                                         surface_points: torch.Tensor,
                                         surface_normals: torch.Tensor,
                                         target_response: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Loss to match target frequency response.

        Args:
            surface_points: (N, 3) surface points
            surface_normals: (N, 3) surface normals
            target_response: (n_angles, n_frequencies) target complex response

        Returns:
            loss: scalar tensor
            info: dict with intermediate values
        """
        # Compute actual response
        actual_response = self.simulator.compute_frequency_response(surface_points, surface_normals)

        if target_response is None:
            target_response = self.target_response

        if target_response is None:
            # Default target: maximize angular variation
            # Use angular diversity as implicit target
            loss = -self.simulator.compute_angular_diversity(actual_response)
            info = {
                'actual_response_magnitude': torch.abs(actual_response).mean().item(),
                'using_default_target': True
            }
            return loss, info

        # MSE loss between target and actual (magnitude and phase)
        magnitude_loss = torch.mean((torch.abs(actual_response) - torch.abs(target_response))**2)
        phase_loss = torch.mean((torch.angle(actual_response) - torch.angle(target_response))**2)

        loss = magnitude_loss + 0.1 * phase_loss  # Weight phase less

        info = {
            'magnitude_loss': magnitude_loss.item(),
            'phase_loss': phase_loss.item(),
            'actual_response_magnitude': torch.abs(actual_response).mean().item(),
            'target_response_magnitude': torch.abs(target_response).mean().item(),
        }

        return loss, info

    def compute_angular_diversity_loss(self,
                                        surface_points: torch.Tensor,
                                        surface_normals: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Loss to maximize angular diversity (different angles have different responses).

        Args:
            surface_points: (N, 3) surface points
            surface_normals: (N, 3) surface normals

        Returns:
            loss: scalar tensor (negative diversity, to minimize)
            info: dict with intermediate values
        """
        # Compute response
        response = self.simulator.compute_frequency_response(surface_points, surface_normals)

        # Diversity metric
        diversity = self.simulator.compute_angular_diversity(response)

        # Negative because we want to maximize diversity (minimize negative diversity)
        loss = -diversity

        info = {
            'angular_diversity': diversity.item(),
            'response_std': torch.abs(response).std().item(),
        }

        return loss, info

    def compute_directional_selectivity_loss(self,
                                              surface_points: torch.Tensor,
                                              surface_normals: torch.Tensor,
                                              target_angle_idx: int = 0) -> Tuple[torch.Tensor, Dict]:
        """
        Loss to maximize directional selectivity (target angle vs others).

        Args:
            surface_points: (N, 3) surface points
            surface_normals: (N, 3) surface normals
            target_angle_idx: Index of target angle

        Returns:
            loss: scalar tensor (negative selectivity)
            info: dict with intermediate values
        """
        # Compute response
        response = self.simulator.compute_frequency_response(surface_points, surface_normals)

        # Selectivity metric
        selectivity = self.simulator.compute_directional_selectivity(response, target_angle_idx)

        # Negative because we want to maximize selectivity
        loss = -selectivity

        info = {
            'directional_selectivity': selectivity.item(),
        }

        return loss, info

    def compute_smoothness_vs_sharpness_loss(self,
                                              surface_points: torch.Tensor,
                                              curvatures: torch.Tensor,
                                              target_sharpness: float = 0.5) -> Tuple[torch.Tensor, Dict]:
        """
        Balance between smooth surfaces and sharp features for diffraction.

        Args:
            surface_points: (N, 3) surface points
            curvatures: (N,) curvature values
            target_sharpness: 0=smooth, 1=sharp

        Returns:
            loss: scalar tensor
            info: dict with intermediate values
        """
        # Variance of curvature indicates feature sharpness
        curvature_var = torch.var(torch.abs(curvatures))

        # Target variance based on desired sharpness
        target_var = target_sharpness * 100.0  # Scale factor

        loss = (curvature_var - target_var)**2

        info = {
            'curvature_variance': curvature_var.item(),
            'target_variance': target_var,
            'mean_curvature': torch.abs(curvatures).mean().item(),
        }

        return loss, info

    def compute_hole_size_constraint_loss(self,
                                           surface_points: torch.Tensor,
                                           min_hole_diameter: float = 0.0005,
                                           max_hole_diameter: float = 0.002) -> Tuple[torch.Tensor, Dict]:
        """
        Constrain hole sizes to be within manufacturing limits.

        This is a soft constraint that penalizes holes outside the size range.

        Args:
            surface_points: (N, 3) surface points
            min_hole_diameter: Minimum hole diameter (m)
            max_hole_diameter: Maximum hole diameter (m)

        Returns:
            loss: scalar tensor
            info: dict with intermediate values
        """
        # This is a placeholder - actual implementation would need to
        # identify holes and measure their sizes from the implicit surface

        # For now, use a regularization based on point density
        # Higher density = smaller features
        if len(surface_points) < 10:
            return torch.tensor(0.0, device=surface_points.device), {}

        # Estimate local feature size from nearest neighbor distances
        # Sample subset for efficiency
        n_samples = min(1000, len(surface_points))
        indices = torch.randperm(len(surface_points))[:n_samples]
        sampled_points = surface_points[indices]

        # Compute pairwise distances (expensive, but only on subset)
        dists = torch.cdist(sampled_points, sampled_points)
        dists = dists + torch.eye(n_samples, device=dists.device) * 1e6  # Exclude self

        # Minimum distance to nearest neighbor
        min_dists = dists.min(dim=1)[0]
        mean_min_dist = min_dists.mean()

        # Feature size should be related to hole diameter
        # Penalize if mean spacing is outside expected range
        target_spacing = (min_hole_diameter + max_hole_diameter) / 4.0  # Rough estimate

        loss = torch.abs(mean_min_dist - target_spacing) * 10.0

        info = {
            'mean_nearest_neighbor_dist': mean_min_dist.item(),
            'target_spacing': target_spacing,
        }

        return loss, info


def create_target_response_frontal_bias(n_angles: int = 7,
                                         n_frequencies: int = 5,
                                         device: str = 'cuda') -> torch.Tensor:
    """
    Create a simple target response with frontal bias.

    Front (0°) should have strong response, sides should be attenuated.

    Args:
        n_angles: Number of angles
        n_frequencies: Number of frequencies
        device: Device for tensor

    Returns:
        target: (n_angles, n_frequencies) complex tensor
    """
    target = torch.zeros(n_angles, n_frequencies, dtype=torch.complex64, device=device)

    # Frontal angle gets highest response
    target[0, :] = 1.0 + 0.0j

    # Side angles get attenuated response with phase shifts
    for i in range(1, n_angles):
        attenuation = 0.3 + 0.5 * np.cos(i * np.pi / n_angles)
        phase_shift = i * np.pi / (2 * n_angles)
        target[i, :] = attenuation * torch.exp(1j * torch.tensor(phase_shift))

    # Add frequency-dependent variation
    for j in range(n_frequencies):
        freq_factor = 0.8 + 0.2 * (j / n_frequencies)
        target[:, j] *= freq_factor

    return target


def create_target_response_omnidirectional(n_angles: int = 7,
                                            n_frequencies: int = 5,
                                            device: str = 'cuda') -> torch.Tensor:
    """
    Create an omnidirectional target response (all angles equal).

    Args:
        n_angles: Number of angles
        n_frequencies: Number of frequencies
        device: Device for tensor

    Returns:
        target: (n_angles, n_frequencies) complex tensor
    """
    return torch.ones(n_angles, n_frequencies, dtype=torch.complex64, device=device)
