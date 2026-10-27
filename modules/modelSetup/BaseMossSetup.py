from abc import ABCMeta
from random import Random
import os
import json
import re
import hashlib
from pathlib import Path
from PIL import Image

import modules.util.multi_gpu_util as multi
from modules.model.MossModel import MossModel, MossModelEmbedding
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_moss_transformer,
    enable_checkpointing_for_moss_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig, TrainEmbeddingConfig
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor
import torch.nn.functional as F


class BaseMossSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "attn-mlp": ["proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"],
        "attn-only": ["proj.q", "proj.k", "proj.v", "proj.o"],
        "blocks": ["layers"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: MossModel,
            config: TrainConfig,
    ):
        super().setup_optimizations(model, config)
        self._setup_model_part(model, config, "transformer", config.transformer, enable_checkpointing_for_moss_transformer)
        self._setup_model_part(model, config, "text_encoder", config.text_encoder, enable_checkpointing_for_moss_encoder_layers, disable_fp16_autocast=True)
        self._setup_model_part(model, config, "vae", config.vae)

        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=False)

    def predict(
            self,
            model: MossModel,
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
                    scaled_noisy_latent_image,
                    latent_noise,
                    config,
                    generator,
                    rand
                )

            latent_input = scaled_noisy_latent_image

            if torch.all(text_attention_mask):
                text_attention_mask = None

            # the MOSS transformer consumes 4D latents at full spatial resolution, a raw 0-1000
            # timestep, and the single last text hidden state - no packing, no position ids
            predicted_flow = model.transformer(
                hidden_states=latent_input.to(dtype=model.train_dtype.torch_dtype()),
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                timestep=timestep,
                encoder_attention_mask=text_attention_mask,
                return_dict=False,
            )[0]

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
            model: MossModel,
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
            model: MossModel,
            embedding_config: TrainEmbeddingConfig,
            hidden_dim: int,
            train_dtype: torch.dtype,
    ) -> Tensor:
        model.materialize_only("text_encoder")

        with torch.no_grad():
            encoded, _ = model.encode_text(
                text=embedding_config.initial_embedding_text,
                train_device=self.train_device,
                apply_append_tokens=False,
            )
            vector = encoded[0].detach()

        token_count = embedding_config.token_count if embedding_config.token_count is not None else vector.shape[0]
        if vector.shape[0] < token_count:
            # pad with the encoding of '*' up to the requested token count
            with torch.no_grad():
                pad_encoded, _ = model.encode_text(
                    text="*",
                    train_device=self.train_device,
                    apply_append_tokens=False,
                )
            pad_vector = pad_encoded[0].detach()
            vector = torch.cat([vector, pad_vector.expand(token_count - vector.shape[0], -1)], dim=0)
        else:
            vector = vector[:token_count]

        result = vector.to(device=self.train_device, dtype=train_dtype).requires_grad_(True)

        model.evict("text_encoder")

        return result

    def _create_append_token(
            self,
            model: MossModel,
            config: TrainConfig,
            embedding_config: TrainEmbeddingConfig,
            hidden_dim: int,
            train_dtype: torch.dtype,
    ) -> Tensor:
        # Pick the initialization strategy: vision-based (from the concept images) when enabled, else text-based.
        if embedding_config.init_from_images:
            return self._create_new_append_token_from_images(model, config, embedding_config, hidden_dim, train_dtype)
        return self._create_new_append_token(model, embedding_config, hidden_dim, train_dtype)

    def _create_new_append_token_from_images(
            self,
            model: MossModel,
            config: TrainConfig,
            embedding_config: TrainEmbeddingConfig,
            hidden_dim: int,
            train_dtype: torch.dtype,
    ) -> Tensor:
        """Initialize the append tokens from the concept images, using the text encoder's vision tower.

        Every enabled concept image whose caption contains the embedding's placeholder is run through the
        vision encoder; the per-image visual features are resampled to the requested token count and averaged,
        so the resulting tokens carry the aggregate appearance of the whole concept. The result is cached with
        the same two-level (memory + disk) scheme as encode_text_cached, so the images are only encoded once.
        """
        # target resolution scale: the smallest component of the configured resolution string
        res_numbers = [int(x) for x in re.split(r'\D', config.resolution) if x.strip() != '']
        target_resolution = min(res_numbers) if res_numbers else 512
        token_count = embedding_config.token_count or 1

        image_entries = self._gather_embedding_images(config, embedding_config)
        if not image_entries:
            print(f"Vision init: no images found for placeholder {embedding_config.placeholder!r}; falling back to text init.")
            return self._create_new_append_token(model, embedding_config, hidden_dim, train_dtype)

        if getattr(self, "_vision_feature_cache", None) is None:
            self._vision_feature_cache = {}
        cache_key = self.__vision_feature_cache_key(config, embedding_config, image_entries, target_resolution, token_count)
        cached = self._vision_feature_cache.get(cache_key)
        if cached is not None:
            return cached.to(device=self.train_device, dtype=train_dtype).requires_grad_(True)

        disk_path = self.__vision_feature_disk_path(config, cache_key)
        if disk_path is not None and os.path.isfile(disk_path):
            vector = torch.load(disk_path, map_location=self.train_device, weights_only=True)
            vector = vector.to(device=self.train_device, dtype=train_dtype).requires_grad_(True)
            self._vision_feature_cache[cache_key] = vector
            return vector

        processor = self.__get_image_processor(config)
        if processor is None:
            print(f"Vision init: image processor unavailable for placeholder {embedding_config.placeholder!r}; falling back to text init.")
            return self._create_new_append_token(model, embedding_config, hidden_dim, train_dtype)
        model.materialize_only("text_encoder")
        text_encoder_device = model.text_encoder.device
        pooled_list = []
        try:
            for image_path, _caption in image_entries:
                try:
                    with Image.open(image_path) as pil:
                        pil = pil.convert("RGB")
                        inputs = processor(images=pil)
                        pixel_values = inputs["pixel_values"].to(device=text_encoder_device)
                        image_grid_thw = inputs["image_grid_thw"].to(device=text_encoder_device)
                        with torch.no_grad():
                            features = model.text_encoder.get_image_features(pixel_values, image_grid_thw)
                        feat = features[0]  # (num_visual_tokens, out_hidden_size)
                        pooled_list.append(self.__pool_to_token_count(feat, token_count))
                except Exception as e:
                    print(f"Vision init: failed to encode image {image_path}: {e}")
        finally:
            model.evict("text_encoder")

        if not pooled_list:
            print(f"Vision init: no images could be encoded for placeholder {embedding_config.placeholder!r}; falling back to text init.")
            return self._create_new_append_token(model, embedding_config, hidden_dim, train_dtype)

        vector = torch.stack(pooled_list, dim=0).mean(dim=0).to(device=self.train_device, dtype=train_dtype).requires_grad_(True)
        self._vision_feature_cache[cache_key] = vector
        if disk_path is not None:
            os.makedirs(os.path.dirname(disk_path), exist_ok=True)
            tmp_path = disk_path + ".tmp"
            torch.save(vector.detach(), tmp_path)
            os.replace(tmp_path, disk_path)
        return vector

    def _gather_embedding_images(self, config: TrainConfig, embedding_config: TrainEmbeddingConfig) -> list[tuple[str, str]]:
        """Collect (image_path, caption) for every enabled concept image whose caption contains the placeholder."""
        from modules.util import path_util
        image_exts = path_util.supported_image_extensions()
        if config.concepts is None:
            with open(config.concept_file_name, 'r') as f:
                concepts = json.load(f)
                for i in range(len(concepts)):
                    concepts[i] = ConceptConfig.default_values().from_dict(concepts[i])
                config.concepts = concepts
        entries = []
        for concept in config.concepts:
            if not concept.enabled:
                continue
            concept_path = Path(concept.path)
            if not concept_path.is_dir():
                continue
            if concept.include_subdirectories:
                image_files = [p for p in concept_path.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]
            else:
                image_files = [p for p in concept_path.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
            for image_file in image_files:
                caption = self.__concept_image_caption(concept, image_file)
                if embedding_config.placeholder and embedding_config.placeholder in caption:
                    entries.append((str(image_file), caption))
        return entries

    def __concept_image_caption(self, concept, image_file: Path) -> str:
        prompt_source = concept.text.prompt_source
        if prompt_source == "filename":
            return image_file.stem
        if prompt_source == "concept":
            prompt_path = concept.text.prompt_path
            if prompt_path:
                caption_file = Path(prompt_path) / (image_file.stem + ".txt")
                if caption_file.is_file():
                    return caption_file.read_text(encoding="utf-8").strip()
            return image_file.stem
        # "sample" (default): a .txt file next to the image
        caption_file = image_file.with_suffix(".txt")
        if caption_file.is_file():
            return caption_file.read_text(encoding="utf-8").strip()
        return image_file.stem

    def __pool_to_token_count(self, feat: Tensor, token_count: int) -> Tensor:
        """Resample (num_tokens, dim) visual features to exactly token_count tokens via linear interpolation.

        Works for every case: token_count == 1 (mean), token_count > num_tokens (upsample), and
        token_count < num_tokens (downsample). A single input token is broadcast rather than interpolated.
        """
        num_tokens, dim = feat.shape
        if num_tokens == 1:
            return feat.expand(token_count, dim).contiguous()
        if num_tokens == token_count:
            return feat
        x = feat.unsqueeze(0).transpose(1, 2)  # (1, dim, num_tokens)
        x = F.interpolate(x, size=(token_count,), mode="linear", align_corners=False)
        return x.transpose(1, 2).squeeze(0)  # (token_count, dim)

    def __vision_feature_cache_key(self, config: TrainConfig, embedding_config, image_entries, target_resolution, token_count) -> str:
        image_sig = hashlib.sha256(
            "".join(f"{p}|{c}\n" for p, c in sorted(image_entries)).encode("utf-8")
        ).hexdigest()
        payload = "\x00".join([
            config.model_type.name,
            embedding_config.placeholder,
            str(target_resolution),
            str(token_count),
            image_sig,
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __vision_feature_disk_path(self, config: TrainConfig, cache_key: str) -> str | None:
        if not config.cache_dir:
            return None
        return os.path.join(config.cache_dir, "vision_embeddings", f"{cache_key}.pt")

    def __get_image_processor(self, config: TrainConfig):
        if getattr(self, "_image_processor", None) is None:
            from transformers import AutoImageProcessor
            base_model = config.model_names().base_model
            try:
                self._image_processor = AutoImageProcessor.from_pretrained(base_model, subfolder="text_encoder")
            except Exception as e:
                print(f"Vision init: AutoImageProcessor failed for {base_model!r} ({e}); falling back to a Qwen2-VL image processor built from the text encoder vision config.")
                self._image_processor = self.__build_qwen_image_processor(base_model)
        return self._image_processor

    def __build_qwen_image_processor(self, base_model: str):
        """Build a Qwen2-VL image processor from the text encoder's vision config.

        The MOSS-Image text encoder is a Qwen3.5-VL model, but the published checkpoint ships no
        preprocessor_config.json, so AutoImageProcessor cannot load one. qwen3_5 registers
        Qwen2VLImageProcessor in this transformers build, which shares the vision tower's
        patch/merge geometry, so we construct one directly from the vision_config.
        """
        try:
            vision = {}
            cfg_path = os.path.join(base_model, "text_encoder", "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r") as f:
                    vision = json.load(f).get("vision_config", {})
            patch_size = vision.get("patch_size", 16)
            temporal_patch_size = vision.get("temporal_patch_size", 2)
            merge_size = vision.get("spatial_merge_size", vision.get("merge_size", 2))
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
            return Qwen2VLImageProcessor(
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
                size={"shortest_edge": 65536, "longest_edge": 16777216},
            )
        except Exception as e:
            print(f"Vision init: could not build a Qwen image processor ({e})")
            return None

    def _setup_embeddings(
            self,
            model: MossModel,
            config: TrainConfig,
    ):
        hidden_dim = model.transformer.config.text_dim if model.transformer is not None else 4096
        train_dtype = model.train_dtype.torch_dtype()

        additional_embeddings = []
        for embedding_config in config.all_embedding_configs():
            embedding_state = model.embedding_state_dicts.get(embedding_config.uuid, None)
            token_count = embedding_config.token_count or 1

            if embedding_state is None:
                append_token = self._create_append_token(model, config, embedding_config, hidden_dim, train_dtype)
            else:
                embedding_state = embedding_state.get("moss", None)

                if embedding_state is not None:
                    append_token = embedding_state.to(
                        dtype=train_dtype,
                        device=self.train_device,
                    ).detach().requires_grad_(True)
                else:
                    append_token = self._create_append_token(model, config, embedding_config, hidden_dim, train_dtype)

            embedding = MossModelEmbedding(
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
            model: MossModel,
            config: TrainConfig,
    ):
        for embedding, embedding_config in zip(model.all_embeddings(),
                                               config.all_embedding_configs(), strict=True):
            train_embedding = \
                embedding_config.train \
                and config.text_encoder.train_embedding \
                and not self.stop_embedding_training_elapsed(embedding_config, model.train_progress)
            model.append_tokens[embedding.uuid].requires_grad_(train_embedding)

    def prepare_text_caching(self, model: MossModel, config: TrainConfig):
        if not config.train_text_encoder_or_embedding():
            model.materialize_only("text_encoder")

        model.eval()
