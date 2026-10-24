from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.Krea2Model import Krea2Model, Krea2ModelEmbedding, TEXT_ENCODER_SELECT_LAYERS
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_krea2_transformer,
    enable_checkpointing_for_qwen3vl_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

from diffusers import Krea2Pipeline


class BaseKrea2Setup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "attn-mlp": ["attn", "ff"],
        "attn-only": ["attn"],
        "blocks": ["transformer_blocks"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: Krea2Model,
            config: TrainConfig,
    ):
        super().setup_optimizations(model, config)
        self._setup_model_part(model, config, "transformer", config.transformer, enable_checkpointing_for_krea2_transformer)
        self._setup_model_part(model, config, "text_encoder", config.text_encoder, enable_checkpointing_for_qwen3vl_encoder_layers, disable_fp16_autocast=True)
        self._setup_model_part(model, config, "vae", config.vae)

        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=True)

    def predict(
            self,
            model: Krea2Model,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            text_encoder_output, text_attention_mask = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_output=batch.get('text_encoder_hidden_state'),
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
                caption_texts=batch.get('prompt'),
            )

            if config.cep_gamma > 0 and not deterministic:
                text_encoder_output = self._apply_conditional_embedding_perturbation(
                    text_encoder_output, config.cep_gamma, generator
                )

            latent_image = batch['latent_image']
            scaled_latent_image = model.scale_latents(latent_image)
            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift = shift if config.dynamic_timestep_shifting else config.timestep_shift,
                sigmas=model.noise_scheduler.sigmas,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            if not deterministic:
                scaled_noisy_latent_image, latent_noise = self._apply_ciop(
                    scaled_noisy_latent_image, latent_noise, config, generator, rand
                )

            latent_input = scaled_noisy_latent_image
            packed_latent_input = model.pack_latents(latent_input)

            # position ids: text tokens at origin, image tokens at latent-grid coords (patch_size = 2)
            text_seq_len = text_encoder_output.shape[1]
            grid_height = latent_input.shape[-2] // 2
            grid_width = latent_input.shape[-1] // 2
            position_ids = Krea2Pipeline.prepare_position_ids(
                text_seq_len, grid_height, grid_width, self.train_device
            )

            if torch.all(text_attention_mask):
                text_attention_mask = None

            packed_predicted_flow = model.transformer(
                hidden_states=packed_latent_input.to(dtype=model.train_dtype.torch_dtype()),
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep / 1000,
                position_ids=position_ids,
                encoder_attention_mask=text_attention_mask,
                return_dict=False,
            )[0]

            predicted_flow = model.unpack_latents(
                packed_predicted_flow,
                height=latent_input.shape[-2],
                width=latent_input.shape[-1],
            )

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow,
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens("7-prompt", batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: Krea2Model,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

    def _create_new_append_token(
            self,
            model: Krea2Model,
            embedding_config: TrainEmbeddingConfig,
            append_dim: int,
            train_dtype: torch.dtype,
    ) -> Tensor:
        model.materialize_only("text_encoder")

        with torch.no_grad():
            encoded, _ = model.encode_text(
                text=embedding_config.initial_embedding_text,
                train_device=self.train_device,
                apply_append_tokens=False,
            )
            # flatten (T, L, H) -> (T, L*H) to match the flat append token layout
            vector = encoded[0].detach().flatten(1)

        token_count = embedding_config.token_count if embedding_config.token_count is not None else vector.shape[0]
        if vector.shape[0] < token_count:
            # pad with the encoding of '*' up to the requested token count
            with torch.no_grad():
                pad_encoded, _ = model.encode_text(
                    text="*",
                    train_device=self.train_device,
                    apply_append_tokens=False,
                )
            pad_vector = pad_encoded[0].detach().flatten(1)
            vector = torch.cat([vector, pad_vector.expand(token_count - vector.shape[0], -1)], dim=0)
        else:
            vector = vector[:token_count]

        result = vector.to(device=self.train_device, dtype=train_dtype).requires_grad_(True)

        model.evict("text_encoder")

        return result

    def _setup_embeddings(
            self,
            model: Krea2Model,
            config: TrainConfig,
    ):
        hidden_dim = model.transformer.config.text_hidden_dim if model.transformer is not None else 2560
        # encode_text works on the flattened (B, T, L*H) text output, so the append
        # token is stored flat as (token_count, num_layers * hidden_dim)
        append_dim = len(TEXT_ENCODER_SELECT_LAYERS) * hidden_dim
        train_dtype = model.train_dtype.torch_dtype()

        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            token_count = embedding_config.token_count or 1

            if embedding_state is None:
                append_token = self._create_new_append_token(model, embedding_config, append_dim, train_dtype)
            else:
                embedding_state = embedding_state.get("krea2", None)

                if embedding_state is not None:
                    append_token = embedding_state.to(
                        dtype=train_dtype,
                        device=self.train_device,
                    ).detach().requires_grad_(True)
                else:
                    append_token = self._create_new_append_token(model, embedding_config, append_dim, train_dtype)

            embedding = Krea2ModelEmbedding(
                embedding_config.uuid,
                embedding_config.placeholder,
                token_count,
            )
            embedding.append_token = append_token
            embedding.vector = append_token  # alias so BaseModelEmbedding.append_embed_token can gate on the trigger word
            model.append_tokens[embedding_config.uuid] = append_token

            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding)

        model.additional_embeddings = additional_embeddings

    def _setup_embeddings_requires_grad(
            self,
            model: Krea2Model,
            config: TrainConfig,
    ):
        for embedding, embedding_config in zip(model.all_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding = \
                embedding_config.train \
                and config.text_encoder.train_embedding \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            model.append_tokens[embedding.uuid].requires_grad_(train_embedding)

    def prepare_text_caching(self, model: Krea2Model, config: TrainConfig):
        if not config.train_text_encoder_or_embedding():
            model.materialize_only("text_encoder")

        model.eval()
