"""
Loss calculator for acoustic stencil optimization.

Extends the base loss calculator with acoustic-specific losses.
"""

import torch
import logging
from typing import Dict, Tuple

from train.train_utils.loss_calculator_base import LossCalculatorBase
from train.losses_acoustic import AcousticLosses, create_target_response_frontal_bias
from train.train_utils.loss_keys import LossKey


class LossCalculatorAcoustic(LossCalculatorBase):
    """
    Loss calculator with acoustic simulation for Owlet stencil design.
    """

    def __init__(self, config: dict, problem, **kwargs):
        super().__init__(config, problem, **kwargs)

        self.logger = logging.getLogger('LossCalculatorAcoustic')

        # Extract acoustic configuration
        acoustic_config = config.get('acoustic', {})

        self.frequencies = acoustic_config.get('target_frequencies', [500, 1000, 2000, 4000, 8000])
        self.angles = acoustic_config.get('target_angles', [0, 30, 60, 90, 120, 150, 180])

        # Create target response if specified
        target_type = acoustic_config.get('target_response_type', 'frontal_bias')
        if target_type == 'frontal_bias':
            self.target_response = create_target_response_frontal_bias(
                n_angles=len(self.angles),
                n_frequencies=len(self.frequencies),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.target_response = None

        # Initialize acoustic loss module
        self.acoustic_losses = AcousticLosses(
            frequencies=self.frequencies,
            angles=self.angles,
            target_response=self.target_response,
            stencil_radius=problem.stencil_radius,
            stencil_height=problem.stencil_height
        )

        # Loss weights from config
        self.lambda_freq_response = config.get('lambda_freq_response', 0.0)
        self.lambda_angular_div = config.get('lambda_angular_div', 0.0)
        self.lambda_directional = config.get('lambda_directional', 0.0)
        self.lambda_hole_size = config.get('lambda_hole_size', 0.0)

        self.logger.info(f'Acoustic losses initialized: freq_response={self.lambda_freq_response}, '
                         f'angular_div={self.lambda_angular_div}, directional={self.lambda_directional}')

    def compute_acoustic_losses(self,
                                 surface_points: torch.Tensor,
                                 surface_normals: torch.Tensor,
                                 curvatures: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute all acoustic losses.

        Args:
            surface_points: (N, 3) surface points
            surface_normals: (N, 3) surface normals
            curvatures: (N,) curvature values (optional)

        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual losses
        """
        total_loss = torch.tensor(0.0, device=surface_points.device)
        loss_dict = {}

        # 1. Frequency response loss
        if self.lambda_freq_response > 0:
            freq_loss, freq_info = self.acoustic_losses.compute_frequency_response_loss(
                surface_points, surface_normals
            )
            total_loss = total_loss + self.lambda_freq_response * freq_loss
            loss_dict[LossKey.FREQ_RESPONSE] = freq_loss.item()
            loss_dict.update({f'freq_response_{k}': v for k, v in freq_info.items()})

        # 2. Angular diversity loss
        if self.lambda_angular_div > 0:
            div_loss, div_info = self.acoustic_losses.compute_angular_diversity_loss(
                surface_points, surface_normals
            )
            total_loss = total_loss + self.lambda_angular_div * div_loss
            loss_dict[LossKey.ANGULAR_DIV] = div_loss.item()
            loss_dict.update({f'angular_div_{k}': v for k, v in div_info.items()})

        # 3. Directional selectivity loss
        if self.lambda_directional > 0:
            dir_loss, dir_info = self.acoustic_losses.compute_directional_selectivity_loss(
                surface_points, surface_normals
            )
            total_loss = total_loss + self.lambda_directional * dir_loss
            loss_dict[LossKey.DIRECTIONAL] = dir_loss.item()
            loss_dict.update({f'directional_{k}': v for k, v in dir_info.items()})

        # 4. Hole size constraint
        if self.lambda_hole_size > 0:
            hole_loss, hole_info = self.acoustic_losses.compute_hole_size_constraint_loss(
                surface_points
            )
            total_loss = total_loss + self.lambda_hole_size * hole_loss
            loss_dict[LossKey.HOLE_SIZE] = hole_loss.item()
            loss_dict.update({f'hole_size_{k}': v for k, v in hole_info.items()})

        return total_loss, loss_dict

    def get_acoustic_info_for_logging(self) -> Dict:
        """Get acoustic configuration info for logging."""
        return {
            'frequencies_Hz': self.frequencies,
            'angles_deg': self.angles,
            'lambda_freq_response': self.lambda_freq_response,
            'lambda_angular_div': self.lambda_angular_div,
            'lambda_directional': self.lambda_directional,
        }
