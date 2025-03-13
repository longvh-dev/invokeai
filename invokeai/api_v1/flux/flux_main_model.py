import torch
from typing import List, Optional, Dict, Union
from PIL import Image

from invokeai.app.invocations.model import (
    CLIPField,
    T5EncoderField,
    TransformerField,
    VAEField,
)
from invokeai.backend.model_manager import LoadedModel
from invokeai.backend.flux.modules.transformer import TransformerBase
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    FLUXConditioningInfo,
)

from invokeai.api_v1.flux.flux_model_loader import (
    FluxModelLoader,
    FluxModelLoaderOutput,
)
from invokeai.api_v1.flux.flux_text_encoder import FluxTextEncoder
from invokeai.api_v1.flux.flux_vae_encode import FluxVaeEncode
from invokeai.api_v1.flux.flux_vae_decode import FluxVaeDecode


class FluxMainModel:
    """
    Flux main model that connects all components to build a flux inference pipeline
    with multi-condition support.
    """

    def __init__(self):
        """
        Initialize the Flux main model.

        Args:
            model_name: Name of the FLUX model to load
        """
        self.model_loader = FluxModelLoader()
        # self.model_components = None
        # self.transformer = self.model_loader.transformer
        # self.text_encoder = self.model_loader.t5_encoder
        # self.clip = self.model_loader.clip
        # self.vae = self.model_loader.vae
        # self.max_seq_len = self.model_loader.max_seq_len

    def load_models(self) -> None:
        """Load all required model components"""
        if self.model_components is not None:
            return  # Models already loaded

        # Load model components
        self.model_components = self.model_loader.invoke()

        # Initialize transformer
        transformer_info = self._load_transformer(self.model_components.transformer)
        self.transformer = transformer_info

        # Initialize text encoder
        self.text_encoder = FluxTextEncoder(
            clip=self.model_components.clip, t5_encoder=self.model_components.t5_encoder
        )

        # Initialize VAE encoder and decoder
        vae_info = self._load_vae(self.model_components.vae)
        self.vae_encoder = FluxVaeEncode(vae=vae_info)
        self.vae_decoder = FluxVaeDecode(vae=vae_info)

    def _load_transformer(self, transformer_field: TransformerField) -> LoadedModel:
        """Load the transformer model from the given field"""
        # This would typically use the InvocationContext, but for this implementation
        # we're assuming direct model loading
        # Return a placeholder for the loaded model
        return LoadedModel(None, None, None)

    def _load_vae(self, vae_field: VAEField) -> LoadedModel:
        """Load the VAE model from the given field"""
        # Similarly, this would use InvocationContext in a real implementation
        # Return a placeholder for the loaded model
        return LoadedModel(None, None, None)

    def encode_prompt(self, prompt: str) -> FLUXConditioningInfo:
        """
        Encode a text prompt into conditioning information.

        Args:
            prompt: Text prompt to encode

        Returns:
            FLUXConditioningInfo containing the encoded prompt
        """
        self.load_models()

        # Create conditioning info from the prompt
        t5_embeddings = self.text_encoder._t5_encode(prompt)
        clip_embeddings = self.text_encoder._clip_encode(prompt)

        return FLUXConditioningInfo(
            clip_embeds=clip_embeddings, t5_embeds=t5_embeddings
        )

    def encode_image(self, image: Union[torch.Tensor, Image.Image]) -> torch.Tensor:
        """
        Encode an image into latents.

        Args:
            image: Input image to encode

        Returns:
            Latent representation of the image
        """
        self.load_models()
        return self.vae_encoder.encode(image)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents into an image.

        Args:
            latents: Latent representation to decode

        Returns:
            Decoded image tensor
        """
        self.load_models()
        return self.vae_decoder.decode(latents)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 768,
        height: int = 768,
        seed: Optional[int] = None,
        image_conditioning: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate an image using the FLUX model.

        Args:
            prompt: The text prompt to guide image generation
            negative_prompt: Negative text prompt for guidance
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            width: Output image width
            height: Output image height
            seed: Random seed for reproducibility
            image_conditioning: Optional image conditioning latents
            mask: Optional mask for inpainting/region-based conditioning

        Returns:
            Generated image tensor
        """
        self.load_models()

        # Set up generation parameters
        device = TorchDevice.choose_torch_device()
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=device).manual_seed(seed)

        # Encode the prompt
        positive_conditioning = self.encode_prompt(prompt)
        negative_conditioning = (
            self.encode_prompt(negative_prompt) if negative_prompt else None
        )

        # Create placeholder for the actual generation step
        # In a real implementation, this would use the transformer model
        # to perform the denoising diffusion process
        latents_shape = (1, 4, height // 8, width // 8)
        latents = torch.randn(latents_shape, generator=generator, device=device)

        # Decode the latents to produce the final image
        image = self.decode_latents(latents)

        return image

    def __del__(self):
        """Clean up resources when the model is deleted"""
        # Force garbage collection of models when this object is destroyed
        self.transformer = None
        self.text_encoder = None
        self.vae_encoder = None
        self.vae_decoder = None
        self.model_components = None
        TorchDevice.empty_cache()


if __name__ == "__main__":
    # Example usage
    model = FluxMainModel()

    # Generate an image
    result = model.generate(
        prompt="A beautiful sunset over the mountains, high quality, detailed",
        num_inference_steps=30,
        width=768,
        height=768,
    )

    # Convert to PIL image and save
    if isinstance(result, torch.Tensor):
        img_pil = Image.fromarray((127.5 * (result + 1.0)).byte().cpu().numpy())
        img_pil.save("flux_generated_image.png")
        print("Image saved to flux_generated_image.png")
