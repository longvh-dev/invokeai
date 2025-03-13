import einops
import torch
from typing import Optional, Union

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
)
from invokeai.app.invocations.model import VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.modules.autoencoder import AutoEncoder
from invokeai.backend.model_manager import LoadedModel
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    image_resized_to_grid_as_tensor,
)
from invokeai.backend.util.devices import TorchDevice
from PIL import Image


class FluxVaeEncode:
    """Encodes an image into latents."""

    def __init__(self, vae: Optional[LoadedModel] = None):
        self.vae = vae

    @staticmethod
    def vae_encode(vae_info: LoadedModel, image_tensor: torch.Tensor) -> torch.Tensor:
        # TODO(ryand): Expose seed parameter at the invocation level.
        # TODO(ryand): Write a util function for generating random tensors that is consistent across devices / dtypes.
        # There's a starting point in get_noise(...), but it needs to be extracted and generalized. This function
        # should be used for VAE encode sampling.
        generator = torch.Generator(
            device=TorchDevice.choose_torch_device()
        ).manual_seed(0)
        with vae_info as vae:
            assert isinstance(vae, AutoEncoder)
            vae_dtype = next(iter(vae.parameters())).dtype
            image_tensor = image_tensor.to(
                device=TorchDevice.choose_torch_device(), dtype=vae_dtype
            )
            latents = vae.encode(image_tensor, sample=True, generator=generator)
            return latents

    @torch.no_grad()
    def encode(self, image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Encodes an image into latents.

        Args:
            image: PIL Image or tensor in format [B, C, H, W] or [C, H, W]
            vae: Loaded VAE model (optional if set during initialization)
            seed: Random seed for reproducibility in sampling

        Returns:
            Latent tensor
        """
        if isinstance(image, Image.Image):
            image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
            if image_tensor.dim() == 3:
                image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")
        else:
            image_tensor = image

        latents = self.vae_encode(vae_info=self.vae, image_tensor=image_tensor)
        return latents
        # image = context.images.get_pil(self.image.image_name)

        # vae_info = context.models.load(self.vae.vae)

        # image_tensor = image_resized_to_grid_as_tensor(image.convert("RGB"))
        # if image_tensor.dim() == 3:
        #     image_tensor = einops.rearrange(image_tensor, "c h w -> 1 c h w")

        # context.util.signal_progress("Running VAE")
        # latents = self.vae_encode(vae_info=vae_info, image_tensor=image_tensor)

        # latents = latents.to("cpu")
        # name = context.tensors.save(tensor=latents)
        # return LatentsOutput.build(latents_name=name, latents=latents, seed=None)
