from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.StableDiffusionModel import StableDiffusionModel, StableDiffusionModelEmbedding
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
from modules.util.dtype_util import create_autocast_context
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

class BaseStableDiffusionSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupDiffusionMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta,
):

    def __init__(self, train_device: torch.device, temp_device: torch.device, debug_mode: bool):
        super().__init__(train_device, temp_device, debug_mode)

    def setup_optimizations(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.vae.enable_gradient_checkpointing()
            model.unet.enable_gradient_checkpointing()
            enable_checkpointing_for_basic_transformer_blocks(model.unet, config, offload_enabled=False)
            enable_checkpointing_for_clip_encoder_layers(model.text_encoder, config)

        if config.force_circular_padding:
            apply_circular_padding_to_conv2d(model.vae)
            apply_circular_padding_to_conv2d(model.unet)
            if model.unet_lora is not None:
                apply_circular_padding_to_conv2d(model.unet_lora)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().unet,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
            config.weight_dtypes().embedding if config.train_any_embedding() else None,
        ], config.enable_autocast_cache)

        quantize_layers(model.text_encoder, self.train_device, model.train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.unet, self.train_device, model.train_dtype, config)

    def _setup_embeddings(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            if embedding_state is None:
                embedding_state = self._create_new_embedding(
                    model,
                    embedding_config,
                    model.tokenizer,
                    model.text_encoder,
                    lambda text: model.encode_text(
                        text=text,
                        train_device=self.temp_device,
                    )[0][1:],
                )
            else:
                embedding_state = embedding_state.get("emp_params_out", embedding_state.get("emp_params", None))

            embedding_state = embedding_state.to(
                dtype=model.text_encoder.get_input_embeddings().weight.dtype,
                device=self.train_device,
            ).detach()

            embedding = StableDiffusionModelEmbedding(
                embedding_config.uuid,
                embedding_state,
                embedding_config.placeholder,
                embedding_config.is_output_embedding,
            )
            if embedding_config.uuid == config.embedding.uuid:
                model.embedding = embedding
            else:
                additional_embeddings.append(embedding)

        model.additional_embeddings = additional_embeddings

        self._add_embeddings_to_tokenizer(model.tokenizer, model.all_text_encoder_embeddings())

    def _setup_embedding_wrapper(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        model.embedding_wrapper = AdditionalEmbeddingWrapper(
            tokenizer=model.tokenizer,
            orig_module=model.text_encoder.text_model.embeddings.token_embedding,
            embeddings=model.all_text_encoder_embeddings(),
        )
        model.embedding_wrapper.hook_to_module()

    def _setup_embeddings_requires_grad(
            self,
            model: StableDiffusionModel,
            config: TrainConfig,
    ):
        for embedding, embedding_config in zip(model.all_text_encoder_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding = \
                embedding_config.train \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            embedding.requires_grad_(train_embedding)

    def predict(
            self,
            model: StableDiffusionModel,
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

            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch['tokens'],
                text_encoder_layer_skip=config.text_encoder_layer_skip,
                text_encoder_output=batch[
                    'text_encoder_hidden_state'] if not config.train_text_encoder_or_embedding() else None,
                text_encoder_dropout_probability=config.text_encoder.dropout_probability,
            )

            latent_image = batch['latent_image']
            scaled_latent_image = latent_image * vae_scaling_factor

            # Standard path if diff2flow or selective_diff2flow is disabled
            if not config.diff2flow and not config.selective_diff2flow:
                # This is the original standard diffusion training path
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

                if config.model_type.has_depth_input():
                    predicted_latent_noise = model.unet(
                        latent_input.to(dtype=model.train_dtype.torch_dtype()),
                        timestep,
                        text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                        batch['latent_depth'].to(dtype=model.train_dtype.torch_dtype()),
                    ).sample
                else:
                    predicted_latent_noise = model.unet(
                        latent_input.to(dtype=model.train_dtype.torch_dtype()),
                        timestep,
                        text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                    ).sample

                model_output_data = {'predicted': predicted_latent_noise, 'timestep': timestep}
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
                use_flow_mask = torch.ones_like(discrete_timestep, dtype=torch.bool) if not config.selective_diff2flow else (
                            discrete_timestep >= diff2flow_threshold)

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
                if config.model_type.has_depth_input():
                    predicted_from_unet = model.unet(
                        combined_unet_sample.to(dtype=model.train_dtype.torch_dtype()),
                        combined_unet_timestep,
                        text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                        batch['latent_depth'].to(dtype=model.train_dtype.torch_dtype()),
                    ).sample
                else:
                    predicted_from_unet = model.unet(
                        combined_unet_sample.to(dtype=model.train_dtype.torch_dtype()),
                        combined_unet_timestep,
                        text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                    ).sample

                # Pass raw output and context to calculate_loss for conversion
                model_output_data = {
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
            model: StableDiffusionModel,
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

        final_predicted = data['predicted']
        final_target = data['target'].clone()

        # For the flow-based samples, convert their target into the model's native prediction space.
        if use_flow_mask.any():
            flow_indices = torch.where(use_flow_mask)[0]

            flow_context = data['flow_context']
            dm_x = flow_context['dm_x'][flow_indices]
            dm_t = flow_context['dm_t'][flow_indices]

            # This is the ground-truth flow velocity u_t = x_1 - x_0
            u_target_for_flow_samples = final_target[flow_indices]

            # Convert u_target into the model's native epsilon space.
            eps_target = model._df_predict_eps_from_z_and_v(dm_x, dm_t, u_target_for_flow_samples).to(
                dtype=model.train_dtype.torch_dtype())

            if model.noise_scheduler.config.prediction_type == 'v_prediction':
                # Convert the epsilon target to the final v-target.
                v_target = model.noise_scheduler.get_velocity(dm_x, eps_target, dm_t)
                final_target[flow_indices] = v_target.to(dtype=model.train_dtype.torch_dtype())
            elif model.noise_scheduler.config.prediction_type == 'epsilon':
                # The target is already in the correct epsilon space.
                final_target[flow_indices] = eps_target

        loss_data = {
            'predicted': final_predicted,
            'target': final_target,
            'timestep': data['timestep'],
            'loss_type': 'target',
            'prediction_type': data['prediction_type'],
        }

        # The standard loss function can now handle the entire batch uniformly.
        return self._diffusion_losses(
            batch=batch,
            data=loss_data,
            config=config,
            train_device=self.train_device,
            betas=model.noise_scheduler.betas,
        ).mean()
