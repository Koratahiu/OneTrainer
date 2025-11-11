
from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.StableDiffusionXLModel import StableDiffusionXLModel, StableDiffusionXLModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupDiffusionMixin import ModelSetupDiffusionMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.module.AdditionalEmbeddingWrapper import AdditionalEmbeddingWrapper
from modules.util.checkpointing_util import (
    enable_checkpointing_for_basic_transformer_blocks,
    enable_checkpointing_for_clip_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.conv_util import apply_circular_padding_to_conv2d
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

PRESETS = {
    "attn-mlp": ["attentions"],
    "attn-only": ["attn"],
    "full": [],
}

class BaseStableDiffusionXLSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupDiffusionMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):

    def setup_optimizations(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.unet.enable_gradient_checkpointing()
            enable_checkpointing_for_basic_transformer_blocks(model.unet, config, offload_enabled=False)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_1, config)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder_2, config)

        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.unet)
            if model.unet_lora is not None:
                apply_circular_padding_to_conv2d(model.unet_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().unet,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().text_encoder_2,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.train_any_embedding() else None,
        ], config.enable_autocast_cache)

        model.vae_autocast_context, model.vae_train_dtype = disable_fp16_autocast_context(
            self.train_device,
            config.train_dtype,
            config.fallback_train_dtype,
            [
                config.weight_dtypes().vae,
            ],
            config.enable_autocast_cache,
        )

        quantize_layers(model.text_encoder_1, self.train_device, model.train_dtype, config)
        quantize_layers(model.text_encoder_2, self.train_device, model.train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.vae_train_dtype, config)
        quantize_layers(model.unet, self.train_device, model.train_dtype, config)

    def _setup_embeddings(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            if embedding_state is None:
                embedding_state_1 = self._create_new_embedding(
                    model,
                    embedding_config,
                    model.tokenizer_1,
                    model.text_encoder_1,
                    lambda text: model.encode_text(
                        text=text,
                        train_device=self.temp_device,
                    )[0][0][1:],
                )

                embedding_state_2 = self._create_new_embedding(
                    model,
                    embedding_config,
                    model.tokenizer_2,
                    model.text_encoder_2,
                    lambda text: model.encode_text(
                        text=text,
                        train_device=self.temp_device,
                    )[1][0][1:],
                )
            else:
                embedding_state_1 = embedding_state.get("clip_l_out", embedding_state.get("clip_l", None))
                embedding_state_2 = embedding_state.get("clip_g_out", embedding_state.get("clip_g", None))

            embedding_state_1 = embedding_state_1.to(
                dtype=model.text_encoder_1.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding_state_2 = embedding_state_2.to(
                dtype=model.text_encoder_2.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding = StableDiffusionXLModelEmbedding(
                embedding_config.uuid,
                embedding_state_1,
                embedding_state_2,
                embedding_config.placeholder,
                embedding_config.is_output_embedding,
            )
            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding)

        model.additional_embeddings = additional_embeddings

        self._add_embeddings_to_tokenizer(model.tokenizer_1, model.all_text_encoder_1_embeddings())
        self._add_embeddings_to_tokenizer(model.tokenizer_2, model.all_text_encoder_2_embeddings())

    def _setup_embedding_wrapper(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        model.embedding_wrapper_1 = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer_1,
            orig_module=model.text_encoder_1.text_model.embeddings.token_embedding,
            embeddings=model.all_text_encoder_1_embeddings(),
        )
        model.embedding_wrapper_2 = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer_2,
            orig_module=model.text_encoder_2.text_model.embeddings.token_embedding,
            embeddings=model.all_text_encoder_2_embeddings(),
        )

        model.embedding_wrapper_1.hook_to_module()
        model.embedding_wrapper_2.hook_to_module()

    def _setup_embeddings_requires_grad(
            self,
            model: StableDiffusionXLModel,
            config: TrainConfig,
    ):
        for embedding, embedding_config in zip(model.all_text_encoder_1_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding_1 = \
                embedding_config.train \
                and config.text_encoder.train_embedding \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            embedding.requires_grad_(train_embedding_1)

        for embedding, embedding_config in zip(model.all_text_encoder_2_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding_2 = \
                embedding_config.train \
                and config.text_encoder_2.train_embedding \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            embedding.requires_grad_(train_embedding_2)

    def predict(
            self,
            model: StableDiffusionXLModel,
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

            vae_scaling_factor = model.vae.config['scaling_factor']

            text_encoder_output, pooled_text_encoder_2_output = model.combine_text_encoder_output(*model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens_1=batch['tokens_1'],
                tokens_2=batch['tokens_2'],
                text_encoder_1_layer_skip=config.text_encoder_layer_skip,
                text_encoder_2_layer_skip=config.text_encoder_2_layer_skip,
                text_encoder_1_output=batch[
                    'text_encoder_1_hidden_state'] if not config.train_text_encoder_or_embedding() else None,
                text_encoder_2_output=batch[
                    'text_encoder_2_hidden_state'] if not config.train_text_encoder_2_or_embedding() else None,
                pooled_text_encoder_2_output=batch[
                    'text_encoder_2_pooled_state'] if not config.train_text_encoder_2_or_embedding() else None,
                text_encoder_1_dropout_probability=config.text_encoder.dropout_probability,
                text_encoder_2_dropout_probability=config.text_encoder_2.dropout_probability,
            ))

            latent_image = batch['latent_image']
            scaled_latent_image = latent_image * vae_scaling_factor

            # original size of the image
            original_height = batch['original_resolution'][0]
            original_width = batch['original_resolution'][1]
            crops_coords_top = batch['crop_offset'][0]
            crops_coords_left = batch['crop_offset'][1]
            target_height = batch['crop_resolution'][0]
            target_width = batch['crop_resolution'][1]

            add_time_ids = torch.stack([
                original_height,
                original_width,
                crops_coords_top,
                crops_coords_left,
                target_height,
                target_width
            ], dim=1)

            add_time_ids = add_time_ids.to(
                dtype=scaled_latent_image.dtype,
                device=scaled_latent_image.device,
            )
            added_cond_kwargs = {"text_embeds": pooled_text_encoder_2_output, "time_ids": add_time_ids}

            # Standard path if diff2flow or selective_diff2flow is disabled
            if not config.diff2flow and not config.selective_diff2flow:
                scaled_latent_conditioning_image = None
                if config.model_type.has_conditioning_image_input():
                    scaled_latent_conditioning_image = batch['latent_conditioning_image'] * vae_scaling_factor

                timestep = self._get_timestep_discrete(
                    model.noise_scheduler.config['num_train_timesteps'],
                    deterministic,
                    generator,
                    scaled_latent_image.shape[0],
                    config,
                )

                latent_noise = self._create_noise(
                    scaled_latent_image,
                    config,
                    generator,
                    timestep,
                    model.noise_scheduler.betas,
                )

                scaled_noisy_latent_image = self._add_noise_discrete(
                    scaled_latent_image,
                    latent_noise,
                    timestep,
                    model.noise_scheduler.betas,
                )

                if config.model_type.has_mask_input() and config.model_type.has_conditioning_image_input():
                    latent_input = torch.concat(
                        [scaled_noisy_latent_image, batch['latent_mask'], scaled_latent_conditioning_image], 1
                    )
                else:
                    latent_input = scaled_noisy_latent_image

                predicted_from_unet = model.unet(
                    sample=latent_input.to(dtype=model.train_dtype.torch_dtype()),
                    timestep=timestep,
                    encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                model_output_data = {'loss_type': 'target', 'predicted': predicted_from_unet, 'timestep': timestep}
                if model.noise_scheduler.config.prediction_type == 'epsilon':
                    model_output_data['target'] = latent_noise
                elif model.noise_scheduler.config.prediction_type == 'v_prediction':
                    model_output_data['target'] = model.noise_scheduler.get_velocity(
                        scaled_latent_image, latent_noise, timestep
                    )
            else:  # diff2flow and the new hybrid mode
                # Get timesteps for the whole batch
                discrete_timestep = self._get_timestep_discrete(
                    model.noise_scheduler.config['num_train_timesteps'],
                    deterministic,
                    generator,
                    scaled_latent_image.shape[0],
                    config,
                )

                # Determine which samples use which training objective
                diff2flow_threshold = config.selective_diff2flow_timesteps
                use_flow_mask = torch.ones_like(discrete_timestep, dtype=torch.bool) if not config.selective_diff2flow else (discrete_timestep >= diff2flow_threshold)

                # Prepare combined tensors for UNet input
                batch_size = scaled_latent_image.shape[0]
                combined_unet_sample = torch.zeros_like(scaled_latent_image)
                combined_unet_timestep = torch.zeros(batch_size, device=config.train_device)

                # Prepare combined tensors for loss calculation
                combined_target = torch.zeros_like(scaled_latent_image)

                # Path 1: Diff2Flow
                if use_flow_mask.any():
                    flow_indices = torch.where(use_flow_mask)[0]
                    flow_timesteps = discrete_timestep[flow_indices]
                    num_train_timesteps = model.noise_scheduler.config['num_train_timesteps']

                    # Reverse OT timesteps to align with diff2flow logic
                    flow_timesteps_reversed = (num_train_timesteps - 1) - flow_timesteps

                    flow_latent_noise = self._create_noise(
                        scaled_latent_image[flow_indices], config, generator, flow_timesteps,
                        model.noise_scheduler.betas,
                    )

                    target_velocity = scaled_latent_image[flow_indices] - flow_latent_noise

                    t_continuous = flow_timesteps_reversed.float() / num_train_timesteps
                    t_reshaped = t_continuous.reshape(-1, *([1] * (scaled_latent_image.dim() - 1)))
                    xt_flow = (1 - t_reshaped) * flow_latent_noise + t_reshaped * scaled_latent_image[flow_indices]

                    dm_t_continuous = model._df_convert_fm_t_to_dm_t(t_continuous)
                    dm_x = model._df_convert_fm_xt_to_dm_xt(xt_flow, t_continuous)

                    combined_unet_sample[flow_indices] = dm_x.to(dtype=model.train_dtype.torch_dtype())
                    combined_unet_timestep[flow_indices] = dm_t_continuous
                    combined_target[flow_indices] = target_velocity

                # Path 2: Standard Diffusion
                if not use_flow_mask.all():
                    std_indices = torch.where(~use_flow_mask)[0]
                    std_timesteps = discrete_timestep[std_indices]

                    std_latent_noise = self._create_noise(
                        scaled_latent_image[std_indices], config, generator, std_timesteps,
                        model.noise_scheduler.betas,
                    )

                    scaled_noisy_latent_image = self._add_noise_discrete(
                        scaled_latent_image[std_indices], std_latent_noise, std_timesteps,
                        model.noise_scheduler.betas,
                    )

                    combined_unet_sample[std_indices] = scaled_noisy_latent_image
                    combined_unet_timestep[std_indices] = std_timesteps.float()

                    if model.noise_scheduler.config.prediction_type == 'epsilon':
                        combined_target[std_indices] = std_latent_noise
                    elif model.noise_scheduler.config.prediction_type == 'v_prediction':
                        combined_target[std_indices] = model.noise_scheduler.get_velocity(
                            scaled_latent_image[std_indices], std_latent_noise, std_timesteps
                        )

                # Run UNet on the combined batch
                predicted_from_unet = model.unet(
                    sample=combined_unet_sample.to(dtype=model.train_dtype.torch_dtype()),
                    timestep=combined_unet_timestep,
                    encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # Pass raw output and context to calculate_loss for conversion
                model_output_data = {
                    'loss_type': 'target',
                    'predicted': predicted_from_unet,
                    'target': combined_target,
                    'timestep': discrete_timestep,
                    'use_flow_mask': use_flow_mask,
                    'flow_context': {
                        'dm_x': combined_unet_sample,
                        'dm_t': combined_unet_timestep
                    }
                }

        model_output_data['prediction_type'] = model.noise_scheduler.config.prediction_type
        return model_output_data

    def calculate_loss(
            self,
            model: StableDiffusionXLModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        use_flow_mask = data.get('use_flow_mask')

        # If no mask, it's a standard batch, use the original loss function
        if use_flow_mask is None:
            return self._diffusion_losses(
                batch=batch,
                data=data,
                config=config,
                train_device=self.train_device,
                betas=model.noise_scheduler.betas,
            ).mean()

        # Hybrid/Flow path: Calculate losses for each part of the batch separately
        batch_size = data['predicted'].shape[0]
        per_sample_losses = torch.zeros(batch_size, device=self.train_device)

        # Part 1: Standard Diffusion Loss for samples below the threshold
        std_indices = torch.where(~use_flow_mask)[0]
        if len(std_indices) > 0:
            std_data = {
                'loss_type': 'target',
                'predicted': data['predicted'][std_indices],
                'target': data['target'][std_indices],
                'timestep': data['timestep'][std_indices],
                'prediction_type': data['prediction_type'],
            }
            std_batch = {k: v[std_indices] for k, v in batch.items() if
                         isinstance(v, Tensor) and v.shape[0] == batch_size}

            loss_std = self._diffusion_losses(
                batch=std_batch,
                data=std_data,
                config=config,
                train_device=self.train_device,
                betas=model.noise_scheduler.betas,
            )
            per_sample_losses[std_indices] = loss_std

        # Part 2: Diff2Flow Loss for samples at or above the threshold
        flow_indices = torch.where(use_flow_mask)[0]
        if len(flow_indices) > 0:
            flow_context = data['flow_context']
            unet_output_flow = data['predicted'][flow_indices]
            target_velocity_flow = data['target'][flow_indices]
            dm_x = flow_context['dm_x'][flow_indices]
            dm_t = flow_context['dm_t'][flow_indices]

            # Convert the UNet's native output (eps/v) to velocity space for the loss calculation
            if model.noise_scheduler.config.prediction_type == 'v_prediction':
                predicted_velocity = model._df_get_vector_field_from_v(unet_output_flow, dm_x, dm_t)
            else:  # 'epsilon'
                predicted_velocity = model._df_get_vector_field_from_eps(unet_output_flow, dm_x, dm_t)

            flow_data = {
                'loss_type': 'target',
                'predicted': predicted_velocity,
                'target': target_velocity_flow,
                'timestep': data['timestep'][flow_indices],
                'prediction_type': data['prediction_type'],
            }
            flow_batch = {k: v[flow_indices] for k, v in batch.items() if
                          isinstance(v, Tensor) and v.shape[0] == batch_size}

            loss_flow = self._flow_matching_losses(
                batch=flow_batch,
                data=flow_data,
                config=config,
                train_device=self.train_device,
            )
            per_sample_losses[flow_indices] = loss_flow

        return per_sample_losses.mean()
