"""
Wrapper functions for acoustic losses to integrate with GINN trainer.

These functions follow the same signature pattern as other GINN losses.
"""

import torch
import torch.nn.functional as F
from train.losses_acoustic import AcousticLosses, create_target_response_frontal_bias
from util.model_utils import tensor_product_xz
from models.point_wrapper import PointWrapper


def _extract_points_tensor(p_surface):
    """Extract actual tensor from PointWrapper if needed."""
    if isinstance(p_surface, PointWrapper):
        return p_surface.data
    return p_surface


def _get_surface_normals(netp, p_surface, z):
    """Helper function to compute surface normals from gradients."""
    # Extract tensor from PointWrapper if needed
    points = _extract_points_tensor(p_surface)

    # Compute gradients at surface points (normals)
    # netp.vf_x gives gradients, which are normals for SDFs
    normals = netp.vf_x(*tensor_product_xz(points, z))
    normals = F.normalize(normals, p=2, dim=-1)  # Normalize to unit vectors
    return normals


def loss_freq_response(z, acoustic_losses, netp, p_surface=None, **kwargs):
    """
    Frequency response loss wrapper.

    Args:
        z: latent codes
        acoustic_losses: AcousticLosses instance
        netp: neural network with partials
        p_surface: surface points (N, 3) or PointWrapper
        **kwargs: additional arguments (ignored)

    Returns:
        loss: scalar tensor
    """
    if p_surface is None:
        return torch.tensor(0.0)

    # Extract tensor from PointWrapper
    surface_points = _extract_points_tensor(p_surface)

    # Get surface normals from gradients
    surface_normals = _get_surface_normals(netp, p_surface, z)

    # Compute loss
    loss, info = acoustic_losses.compute_frequency_response_loss(surface_points, surface_normals)

    return loss


def loss_angular_div(z, acoustic_losses, netp, p_surface=None, **kwargs):
    """
    Angular diversity loss wrapper.

    Args:
        z: latent codes
        acoustic_losses: AcousticLosses instance
        netp: neural network with partials
        p_surface: surface points (N, 3) or PointWrapper
        **kwargs: additional arguments (ignored)

    Returns:
        loss: scalar tensor (non-negative)
    """
    if p_surface is None:
        return torch.tensor(0.0)

    # Extract tensor from PointWrapper
    surface_points = _extract_points_tensor(p_surface)

    # Get surface normals from gradients
    surface_normals = _get_surface_normals(netp, p_surface, z)

    # Compute loss (returns negative diversity)
    loss, info = acoustic_losses.compute_angular_diversity_loss(surface_points, surface_normals)

    # Convert to non-negative by squaring: min(-d) == min(d^2) for maximizing d
    # This ensures the loss is always non-negative while still maximizing diversity
    return loss ** 2


def loss_directional(z, acoustic_losses, netp, p_surface=None, **kwargs):
    """
    Directional selectivity loss wrapper.

    Args:
        z: latent codes
        acoustic_losses: AcousticLosses instance
        netp: neural network with partials
        p_surface: surface points (N, 3) or PointWrapper
        **kwargs: additional arguments (ignored)

    Returns:
        loss: scalar tensor (non-negative)
    """
    if p_surface is None:
        return torch.tensor(0.0)

    # Extract tensor from PointWrapper
    surface_points = _extract_points_tensor(p_surface)

    # Get surface normals from gradients
    surface_normals = _get_surface_normals(netp, p_surface, z)

    # Compute loss (returns negative selectivity)
    loss, info = acoustic_losses.compute_directional_selectivity_loss(surface_points, surface_normals)

    # Convert to non-negative by squaring: min(-s) == min(s^2) for maximizing s
    # This ensures the loss is always non-negative while still maximizing selectivity
    return loss ** 2


def loss_hole_size(z, acoustic_losses, netp, p_surface=None, **kwargs):
    """
    Hole size constraint loss wrapper.

    Args:
        z: latent codes
        acoustic_losses: AcousticLosses instance
        netp: neural network with partials
        p_surface: surface points (N, 3) or PointWrapper
        **kwargs: additional arguments (ignored)

    Returns:
        loss: scalar tensor
    """
    if p_surface is None:
        return torch.tensor(0.0)

    # Extract tensor from PointWrapper
    surface_points = _extract_points_tensor(p_surface)

    # Compute loss
    loss, info = acoustic_losses.compute_hole_size_constraint_loss(surface_points)

    return loss
