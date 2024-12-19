"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional

import torch 
from torch import Tensor, nn

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field
from nerfstudio.field_components import MLP


class SplatfactoenvField(Field):
    """Splatfactoenv Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    # aabb: Tensor

    def __init__(
        self,
        num_env_params: int, 
        implementation: Literal["tcnn", "torch"] = "torch",
        sh_levels: int = 4,
        num_layers: int = 3,
        num_sh_base_layers: int = 3,
        num_sh_rest_layers = 3,
        layer_width: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = MLP(
            in_dim=3+4+3,
            num_layers=num_layers, 
            layer_width=layer_width,
            out_dim=layer_width, 
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
            implementation= implementation
        )

        self.sh_dim = (sh_levels+1)**2

        self.sh_base_head = MLP(
            in_dim=self.encoder.out_dim+num_env_params,
            num_layers=num_sh_base_layers,
            layer_width=layer_width,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
            implementation=implementation
        )

        self.sh_rest_head = MLP(
            in_dim=self.encoder.out_dim+num_env_params,
            num_layers=num_sh_rest_layers, 
            layer_width=layer_width,
            out_dim=(self.sh_dim-1)*3,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation
        )

    def forward(
        self,
        mean,
        quat,
        scale,
        env_params
    ) -> Tensor:
        x = self.encoder(
            torch.cat((mean, quat, scale), dim=1)
        ).float()
        base_color = self.sh_base_head(
            torch.cat((x, env_params), dim=1)
        )
        sh_rest = self.sh_rest_head(
            torch.cat((x,env_params), dim=1)
        )
        sh_coeffs = torch.cat((base_color, sh_rest), dim=1).view(-1, self.sh_dim, 3)
        return sh_coeffs

class SplatfactoenvRGBField(Field):
    """Splatfactoenv Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    # aabb: Tensor

    def __init__(
        self,
        num_env_params: int, 
        implementation: Literal["tcnn", "torch"] = "torch",
        feature_channel: int = 3,
        num_layers: int = 3,
        num_sh_base_layers: int = 3,
        layer_width: int = 256,
    ) -> None:
        super().__init__()

        self.feature_channel = feature_channel

        self.encoder = MLP(
            in_dim=3+4+3,
            num_layers=num_layers, 
            layer_width=layer_width,
            out_dim=layer_width, 
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
            implementation= implementation
        )

        self.rgb_head = MLP(
            in_dim=self.encoder.out_dim+num_env_params,
            num_layers=num_sh_base_layers,
            layer_width=layer_width,
            out_dim=feature_channel,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
            implementation=implementation
        )

    def forward(
        self,
        mean,
        quat,
        scale,
        env_params
    ) -> Tensor:
        x = self.encoder(
            torch.cat((mean, quat, scale), dim=1)
        ).float()
        base_color = self.rgb_head(
            torch.cat((x, env_params), dim=1)
        )

        rgbs = base_color.view(-1, self.feature_channel)
        return rgbs
