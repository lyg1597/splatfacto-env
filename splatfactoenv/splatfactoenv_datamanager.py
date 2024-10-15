"""
Template DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Optional

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
# from splatfactoenv.splatfactoenv_model import SplatfactoModel
from splatfactoenv.splatfactoenv_dataparser import SplatfactoEnvDataParserConfig, SplatfactoEnvDataParser
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager


@dataclass
class SplatfactoEnvDataManagerConfig(FullImageDatamanagerConfig):
    """Splatfactoenv DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: SplatfactoenvDataManager)
    dataparser: AnnotatedDataParserUnion = field(default_factory=SplatfactoEnvDataParserConfig)
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    cache_images: Literal["cpu", "gpu"] = "gpu"
    """Whether to cache images in memory. If "cpu", caches on cpu. If "gpu", caches on device."""
    cache_images_type: Literal["uint8", "float32"] = "float32"
    """The image type returned from manager, caching images in uint8 saves memory"""
    max_thread_workers: Optional[int] = None
    """The maximum number of threads to use for caching images. If None, uses all available threads."""
    train_cameras_sampling_strategy: Literal["random", "fps"] = "random"
    """Specifies which sampling strategy is used to generate train cameras, 'random' means sampling 
    uniformly random without replacement, 'fps' means farthest point sampling which is helpful to reduce the artifacts 
    due to oversampling subsets of cameras that are very close to each other."""
    train_cameras_sampling_seed: int = 42
    """Random seed for sampling train cameras. Fixing seed may help reduce variance of trained models across 
    different runs."""
    fps_reset_every: int = 100
    """The number of iterations before one resets fps sampler repeatly, which is essentially drawing fps_reset_every
    samples from the pool of all training cameras without replacement before a new round of sampling starts."""



class SplatfactoenvDataManager(FullImageDatamanager):
    """Splatfactoenv DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: SplatfactoEnvDataManagerConfig

    def __init__(
        self,
        config: SplatfactoEnvDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""
        image_idx = self.train_unseen_cameras.pop(0)
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = self.sample_train_cameras()

        data = self.cached_train[image_idx]
        data["image"] = data["image"].to(self.device)

        assert len(self.train_cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        camera.metadata['env_params'] = self.train_dataset.metadata['env_params'][image_idx,:]
        return camera, data
