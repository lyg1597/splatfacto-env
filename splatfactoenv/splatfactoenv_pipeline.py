from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch
import torch.distributed as dist
import torchvision.utils as vutils
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import Pipeline

@dataclass
class SplatfactoEnvPipelineConfig(InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SplatfactoEnvPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = field(default_factory=DataManagerConfig)
    """specifies the datamanager config"""
    model: ModelConfig = field(default_factory=ModelConfig)
    """specifies the model config"""


class SplatfactoEnvPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: SplatfactoEnvPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the dataset and get the average.

        Args:
            data_loader: the data loader to iterate over
            image_prefix: prefix to use for the saved image filenames
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            idx = 0
            for camera, batch in data_loader:
                # time this the following line
                inner_start = time()
                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        vutils.save_image(
                            image.permute(2, 0, 1).cpu(), output_path / f"{image_prefix}_{key}_{idx:04d}.png"
                        )

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        self.train()
        return metrics_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Get the average metrics for evaluation images."""

        """Get the average metrics for evaluation images."""
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        image_prefix = "eval"
        return self.get_average_image_metrics(
            self.datamanager.fixed_indices_eval_dataloader, image_prefix, step, output_path, get_std
        )

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params}
