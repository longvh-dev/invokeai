from ast import Load
from typing import Optional

import torch
from einops import rearrange
from PIL import Image

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.flux.modules.autoencoder import AutoEncoder
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.util.devices import TorchDevice


class FluxVaeDecode:
    """Generates an image from latents."""

    # latents: LatentsField = InputField(
    #     description=FieldDescriptions.latents,
    #     input=Input.Connection,
    # )
    # vae: VAEField = InputField(
    #     description=FieldDescriptions.vae,
    #     input=Input.Connection,
    # )

    def __init__(self, vae: Optional[LoadedModel] = None):
        self.vae = vae
        self.latent_scale_factor = LATENT_SCALE_FACTOR

    def _estimate_working_memory(self, latents: torch.Tensor, vae: AutoEncoder) -> int:
        """Estimate the working memory required by the invocation in bytes."""
        # It was found experimentally that the peak working memory scales linearly with the number of pixels and the
        # element size (precision).
        out_h = self.latent_scale_factor * latents.shape[-2]
        out_w = self.latent_scale_factor * latents.shape[-1]
        element_size = next(vae.parameters()).element_size()
        scaling_constant = 1090  # Determined experimentally.
        working_memory = out_h * out_w * element_size * scaling_constant

        # We add a 20% buffer to the working memory estimate to be safe.
        working_memory = working_memory * 1.2
        return int(working_memory)

    # def _vae_decode(self, vae_info: LoadedModel, latents: torch.Tensor) -> Image.Image:
    #     estimated_working_memory = self._estimate_working_memory(
    #         latents, vae_info.model
    #     )
    #     with vae_info.model_on_device(working_mem_bytes=estimated_working_memory) as (
    #         _,
    #         vae,
    #     ):
    #         assert isinstance(vae, AutoEncoder)
    #         vae_dtype = next(iter(vae.parameters())).dtype
    #         latents = latents.to(
    #             device=TorchDevice.choose_torch_device(), dtype=vae_dtype
    #         )
    #         img = vae.decode(latents)

    #     img = img.clamp(-1, 1)
    #     img = rearrange(img[0], "c h w -> h w c")  # noqa: F821
    #     img_pil = Image.fromarray((127.5 * (img + 1.0)).byte().cpu().numpy())
    #     return img_pil

    # @torch.no_grad()
    # def invoke(self, context: InvocationContext) -> ImageOutput:
    #     latents = context.tensors.load(self.latents.latents_name)
    #     vae_info = context.models.load(self.vae.vae)
    #     context.util.signal_progress("Running VAE")
    #     image = self._vae_decode(vae_info=vae_info, latents=latents)

    #     TorchDevice.empty_cache()
    #     image_dto = context.images.save(image=image)
    #     return ImageOutput.build(image_dto)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        estimated_working_memory = self._estimate_working_memory(
            latents, self.vae.model
        )
        with self.vae.model_on_device(working_mem_bytes=estimated_working_memory) as (
            _,
            vae,
        ):
            assert isinstance(vae, AutoEncoder)
            vae_dtype = next(iter(vae.parameters())).dtype
            latents = latents.to(
                device=TorchDevice.choose_torch_device(), dtype=vae_dtype
            )
            img = vae.decode(latents)

        img = img.clamp(-1, 1)
        img = rearrange(img[0], "c h w -> h w c")

        return img
