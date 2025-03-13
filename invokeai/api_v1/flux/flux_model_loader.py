from typing import Literal, List

from invokeai.app.invocations.model import (
    CLIPField,
    ModelIdentifierField,
    T5EncoderField,
    TransformerField,
    VAEField,
)
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.t5_model_identifier import (
    preprocess_t5_encoder_model_identifier,
    preprocess_t5_tokenizer_model_identifier,
)
from invokeai.backend.flux.util import max_seq_lengths
from invokeai.backend.model_manager.config import (
    CheckpointConfigBase,
    SubModelType,
)


from invokeai.backend.util.logging import InvokeAILogger
from invokeai.app.services.model_records.model_records_sql import ModelRecordServiceSQL
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.app.services.config.config_default import get_config
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage

# region init
app_config = get_config()
logger = InvokeAILogger.get_logger()
output_folder = app_config.outputs_path
if output_folder is None:
    raise ValueError("Output folder is not set")

image_files = DiskImageFileStorage(f"{output_folder}/images")
db = init_db(config=app_config, logger=logger, image_files=image_files)
sql = ModelRecordServiceSQL(db=db, logger=logger)


def get_model_identifier_field(model_name: str) -> ModelIdentifierField:
    return ModelIdentifierField.from_config(
        sql.search_by_attr(model_name=model_name)[0]
    )


class FluxModelLoaderOutput:
    """Flux base model loader output"""

    # transformer: TransformerField
    # clip: CLIPField
    # t5_encoder: T5EncoderField
    # vae: VAEField
    # max_seq_len: Literal[256, 512]

    def __init__(
        self,
        transformer: TransformerField,
        clip: CLIPField,
        t5_encoder: T5EncoderField,
        vae: VAEField,
        max_seq_len: Literal[256, 512],
    ):
        self.transformer = transformer
        self.clip = clip
        self.t5_encoder = t5_encoder
        self.vae = vae
        self.max_seq_len = max_seq_len

    def __repr__(self):
        return f"FluxModelLoaderOutput(transformer={self.transformer}, clip={self.clip}, t5_encoder={self.t5_encoder}, vae={self.vae}, max_seq_len={self.max_seq_len})"


class FluxModelLoader:
    """Loads a flux base model, outputting its submodels."""

    model = get_model_identifier_field("FLUX Dev")
    t5_encoder_model = get_model_identifier_field("t5_base_encoder")
    clip_embed_model = get_model_identifier_field("clip-vit-large-patch14")
    vae_model = get_model_identifier_field("FLUX.1-schnell_ae")

    def invoke(self) -> FluxModelLoaderOutput:
        # for key in [
        #     self.model.key,
        #     self.t5_encoder_model.key,
        #     self.clip_embed_model.key,
        #     self.vae_model.key,
        # ]:
        #     if not context.models.exists(key):
        #         raise ValueError(f"Unknown model: {key}")

        transformer = self.model.model_copy(
            update={"submodel_type": SubModelType.Transformer}
        )
        vae = self.vae_model.model_copy(update={"submodel_type": SubModelType.VAE})

        tokenizer = self.clip_embed_model.model_copy(
            update={"submodel_type": SubModelType.Tokenizer}
        )
        clip_encoder = self.clip_embed_model.model_copy(
            update={"submodel_type": SubModelType.TextEncoder}
        )

        tokenizer2 = preprocess_t5_tokenizer_model_identifier(self.t5_encoder_model)
        t5_encoder = preprocess_t5_encoder_model_identifier(self.t5_encoder_model)

        # transformer_config = context.models.get_config(transformer)
        # assert isinstance(transformer_config, CheckpointConfigBase)

        return FluxModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            clip=CLIPField(
                tokenizer=tokenizer,
                text_encoder=clip_encoder,
                loras=[],
                skipped_layers=0,
            ),
            t5_encoder=T5EncoderField(
                tokenizer=tokenizer2, text_encoder=t5_encoder, loras=[]
            ),
            vae=VAEField(vae=vae),
            max_seq_len=512,
        )


if __name__ == "__main__":
    model = FluxModelLoader().invoke()
    # print(model)

    print(model.transformer)
    # print(model.clip)
    # print(model.t5_encoder)
    # print(model.vae)
    # print(model.max_seq_len)
    # print(model.vae_model)
