import hashlib
import math
import os
import re
from contextlib import nullcontext
from random import Random

import torch
import torch.nn.functional as F
from torch import Tensor

from diffusers import AutoencoderKLFlux2, DiffusionPipeline
from transformers import Qwen2Tokenizer, Qwen3_5Model

from modules.model.BaseModel import BaseModel, BaseModelEmbedding
from modules.model.moss.pipeline_moss_image import (
    MOSSImagePipeline,
    MOSSImageScheduler,
    MOSSImageTransformer,
)
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.enum.ModelType import ModelType

# Maximum number of tokens for a chat-templated prompt (template overhead + user content).
PROMPT_MAX_LENGTH = 4096


def format_moss_prompt(tokenizer: Qwen2Tokenizer, prompt: str) -> str:
    """Render a caption through the MOSS chat template (matches the official pipeline's encode_prompt)."""
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


class MossModelEmbedding(BaseModelEmbedding):
    def __init__(
            self,
            uuid: str,
            placeholder: str,
            token_count: int,
    ):
        # appended embedding: the learned vector is appended to the text encoder
        # output (not written into token positions like an output embedding), so it is
        # not an output embedding.
        super().__init__(uuid, placeholder, vector=None, is_output_embedding=False)
        self.token_count = token_count
        self.append_token: Tensor | None = None

        # vision-based init state (set by BaseMossSetup when the config's image_init is on
        # and the embedding is new): the append tokens start from a temporary text-based
        # init and are replaced in place by the vision-tower pooled encoding of the
        # concept's training images once enough of them have been seen in the stream.
        self.image_init_pending: bool = False
        self.image_init_paths: list[str] = []
        self.image_init_target: int | None = None


class MossModel(BaseModel):
    # base model data
    tokenizer: Qwen2Tokenizer | None
    noise_scheduler: MOSSImageScheduler | None
    text_encoder: Qwen3_5Model | None
    vae: AutoencoderKLFlux2 | None
    transformer: MOSSImageTransformer | None

    # autocast context
    text_encoder_autocast_context: torch.autocast | nullcontext

    # persistent lora training data
    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    embedding: MossModelEmbedding | None
    additional_embeddings: list[MossModelEmbedding] | None
    append_tokens: dict[str, Tensor] | None

    def __init__(
            self,
            model_type: ModelType,
    ):
        super().__init__(
            model_type=model_type,
        )

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()

        self.transformer_lora = None
        self.lora_state_dict = None

        self.embedding = None
        self.additional_embeddings = []
        self.append_tokens = {}

    def fusion_groups(self) -> list | None:
        # MOSS fuses the DiT block q/k/v into one qkv Linear per layer; everything else in the
        # transformer -- text_embedder, time_embedding, shared_block_modulation, proj.o, ffn_up_gate,
        # ffn_down, final layers -- already matches the native checkpoint's naming.
        return [
            ("layers.{i}.proj", ["q", "k", "v"], "qkv", "qkv"),
        ]

    def diffusers_to_original(self) -> list | None:
        # MOSS's native (ComfyUI/original) checkpoint is identical to diffusers except for the q/k/v
        # fusion handled by the checkpoint pre-stage (fusion_groups above, proj.q/k/v -> proj.qkv).
        # After that fusion, every name already matches the native layout, so a single catch-all
        # identity pattern covers the body.
        return [
            ("{path}", "{path}"),
        ]

    def create_pipeline(self) -> DiffusionPipeline:
        return MOSSImagePipeline(
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str | list[str] = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
            apply_append_tokens: bool = True,
            caption_texts: list[str] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            # the encoder is causal, so real-token hidden states are identical with or
            # without padding -- pad only to the batch max ('longest') instead of
            # PROMPT_MAX_LENGTH to avoid encoding ~4000 pad tokens per caption.
            tokens_out = self.tokenizer(
                [format_moss_prompt(self.tokenizer, t) for t in text],
                truncation=True,
                padding='longest',
                max_length=PROMPT_MAX_LENGTH,
                return_tensors='pt',
            )
            tokens = tokens_out.input_ids.to(self.text_encoder.device)
            tokens_mask = tokens_out.attention_mask.to(self.text_encoder.device)

        if text_encoder_output is not None:
            text_encoder_output = text_encoder_output.to(device=train_device)
        if tokens_mask is not None:
            tokens_mask = tokens_mask.to(device=train_device)

        if text_encoder_output is None:
            with self.text_encoder_autocast_context:
                outputs = self.text_encoder(
                    tokens,
                    attention_mask=tokens_mask,
                    output_hidden_states=True,
                )
                # single last hidden state → (B, T, H)
                text_encoder_output = outputs.hidden_states[-1]

        if text_encoder_dropout_probability is not None and text_encoder_dropout_probability > 0.0:
            raise NotImplementedError

        # real tokens are a contiguous prefix (padding sits at the end), so gather the
        # unmasked tokens per sample, append append tokens, then re-pad the batch
        seq_lengths = tokens_mask.sum(dim=1)
        pruned_hidden_states = [text_encoder_output[i][tokens_mask[i].bool()] for i in range(text_encoder_output.shape[0])]

        if apply_append_tokens and self.append_tokens:
            active_embeddings = [
                emb for emb in self.all_embeddings()
                if emb.uuid in self.append_tokens
            ]
            if active_embeddings:
                # per-sample trigger gating lives on the embedding module itself
                # (BaseModelEmbedding.append_embed_token): an embedding's append
                # tokens are appended only when that sample's caption contains its
                # trigger word (placeholder). caption_texts=None appends everywhere.
                captions = caption_texts if caption_texts is not None else [None] * len(pruned_hidden_states)
                new_states = []
                new_lengths = []
                for i, h in enumerate(pruned_hidden_states):
                    caption = captions[i] if i < len(captions) else None
                    for emb in active_embeddings:
                        append = emb.append_embed_token(caption, train_device, text_encoder_output.dtype)
                        if append is not None:
                            h = torch.cat([h, append], dim=0)
                    new_states.append(h)
                    new_lengths.append(h.shape[0])
                pruned_hidden_states = new_states
                seq_lengths = torch.tensor(new_lengths, device=train_device, dtype=seq_lengths.dtype)

        text_encoder_output = torch.nn.utils.rnn.pad_sequence(pruned_hidden_states, batch_first=True)
        max_seq_length = text_encoder_output.shape[1]

        # pad to a multiple of 16 to keep cuDNN's fused-attention shape cache hot across steps
        if max_seq_length % 16 != 0 and (seq_lengths != seq_lengths[0]).any():
            padded_length = max_seq_length + (16 - max_seq_length % 16)
            text_encoder_output = F.pad(text_encoder_output, (0, 0, 0, padded_length - max_seq_length))
            max_seq_length = padded_length

        bool_attention_mask = torch.arange(max_seq_length, device=train_device).unsqueeze(0) < seq_lengths.unsqueeze(1)

        return (text_encoder_output, bool_attention_mask)

    def encode_images_vision_pooled(
            self,
            image_paths: list[str],
            token_count: int,
            train_device: torch.device,
            train_dtype: torch.dtype,
    ) -> Tensor | None:
        """Encode images with the text encoder's vision tower and pool the result to
        token_count vectors in the text hidden space.

        Each path is loaded from disk, resized (aspect-preserving) to a multiple of 32 px
        (patch size 16 times the spatial merge of 2, capped at the configured train
        resolution), duplicated along the temporal axis, unfolded into 16x16 patches, and
        passed through get_image_features one image at a time - the per-image tensors are
        freed immediately after encoding so memory never holds more than one image; the
        per-image patch tokens are mean-pooled to one vector, then the stack of vectors is
        linearly interpolated to token_count. Returns None if no image could be loaded.
        """
        import numpy as np
        from PIL import Image

        patch = 16
        multiple = patch * 2  # spatial merge size
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
        # target box from the configured train resolution (e.g. "1024x768" -> width x height)
        res_dims = [int(x.strip()) for x in re.split(r'\D', self.train_config.resolution) if x.strip() != '']
        box_w, box_h = (res_dims[0], res_dims[1]) if len(res_dims) >= 2 else (res_dims[0], res_dims[0])

        def _load_resized(path: str) -> Tensor | None:
            try:
                with Image.open(path) as im:
                    im = im.convert('RGB')
                    w, h = im.size
                    scale = min(box_w / w, box_h / h, 1.0)
                    new_w = max(multiple, int(round(w * scale / multiple)) * multiple)
                    new_h = max(multiple, int(round(h * scale / multiple)) * multiple)
                    im = im.resize((new_w, new_h))
                    arr = np.asarray(im, dtype=np.float32)
            except Exception:
                return None
            x = torch.from_numpy(arr).permute(2, 0, 1) / 255.0
            return (x - mean) / std

        pooled_images = []
        model = self.text_encoder
        self.materialize_only("text_encoder")
        try:
            with torch.no_grad(), self.text_encoder_autocast_context:
                # load, encode, and free one image at a time (streaming from disk)
                for path in image_paths:
                    x = _load_resized(path)
                    if x is None:
                        continue
                    c, h, w = x.shape
                    # duplicate along the temporal axis, then unfold the SPATIAL dims (2, 3) into
                    # (C, 2, ph, pw, patch, patch). The qwen3_5 tower orders its patches BLOCK-MAJOR:
                    # (merged 2x2 block row, block col, intra-block row, intra-block col)
                    xt = x.unsqueeze(1).repeat(1, 2, 1, 1)
                    patches = xt.unfold(2, patch, patch).unfold(3, patch, patch)
                    merged_h, merged_w = h // (patch * 2), w // (patch * 2)
                    patches = patches.reshape(c, 2, merged_h, 2, merged_w, 2, patch, patch) \
                                     .permute(2, 4, 3, 5, 0, 1, 6, 7).reshape(-1, c, 2, patch, patch)
                    pixel_values = patches.to(device=train_device, dtype=torch.float32)
                    grid_thw = torch.tensor([[1, h // patch, w // patch]], device=train_device, dtype=torch.long)
                    outputs = model.get_image_features(pixel_values, grid_thw)
                    pooled_images.extend(feat.mean(dim=0) for feat in outputs.pooler_output)
                    del x, xt, patches, pixel_values, outputs
                    torch.cuda.empty_cache()
        finally:
            self.evict("text_encoder")

        if not pooled_images:
            return None
        stacked = torch.stack(pooled_images, dim=0)  # (num_images, hidden_dim)
        num_images = stacked.shape[0]
        if num_images != token_count:
            if num_images == 1:
                # a single image cannot be resampled; replicate its vector
                stacked = stacked.expand(token_count, -1)
            elif num_images > token_count:
                # downsample: block-average over the full set of image vectors so every
                # token carries the meaning of all images (a pure linear resample would
                # only blend a local window and, at token_count=1, just two images)
                stacked = F.adaptive_avg_pool1d(
                    stacked.unsqueeze(0).transpose(1, 2), output_size=token_count
                ).transpose(1, 2).squeeze(0)
            else:
                # upsample: linearly blend between neighboring image vectors (linear mode
                # resizes the last dimension, so move the image axis to the end)
                stacked = F.interpolate(
                    stacked.unsqueeze(0).transpose(1, 2), size=token_count, mode='linear', align_corners=False
                ).transpose(1, 2).squeeze(0)
        return stacked.to(device=train_device, dtype=train_dtype)

    def encode_images_vision_pooled_cached(
            self,
            image_paths: list[str],
            token_count: int,
            train_device: torch.device,
            train_dtype: torch.dtype,
    ) -> Tensor | None:
        """On-disk cache around encode_images_vision_pooled, mirroring encode_text_cached's
        disk tier. The vision-tower encoding of a fixed image set is expensive (it
        materializes the text encoder), so it is computed once and reused across runs: a
        repeat call for the same image set loads the stored tensor instead of re-encoding.
        The paths are canonicalized (sorted, deduplicated) so the key - and the
        interpolated result - is independent of the order the training stream happened to
        deliver them in, which is what makes a repeat run hit the disk cache."""
        canonical_paths = tuple(sorted(set(image_paths)))
        if not canonical_paths:
            return None

        disk_path = self.__vision_encoding_disk_path(canonical_paths, token_count)
        if disk_path is not None and os.path.isfile(disk_path):
            return torch.load(disk_path, map_location=train_device, weights_only=True).to(
                device=train_device, dtype=train_dtype
            )

        encoding = self.encode_images_vision_pooled(
            image_paths=list(canonical_paths),
            token_count=token_count,
            train_device=train_device,
            train_dtype=train_dtype,
        )
        if encoding is None:
            return None
        if disk_path is not None:
            os.makedirs(os.path.dirname(disk_path), exist_ok=True)
            tmp_path = disk_path + ".tmp"
            torch.save(encoding.to(dtype=torch.float32, device="cpu"), tmp_path)
            os.replace(tmp_path, disk_path)
        return encoding

    def __vision_encoding_disk_key(self, canonical_paths: tuple[str, ...], token_count: int) -> str:
        # ModelType + token_count + resolution + the canonical image set.
        # train_device/train_dtype are transient and excluded; every input that changes the
        # encoding is part of the key.
        resolution = self.train_config.resolution if self.train_config is not None else ''
        key_str = "\x00".join([
            self.model_type.name,
            str(token_count),
            resolution,
            "\n".join(canonical_paths),
        ])
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def __vision_encoding_disk_path(self, canonical_paths: tuple[str, ...], token_count: int) -> str | None:
        # cache_dir/vision_encodings/<key>.pt; None when there is no cache_dir (memory only).
        if self.train_config is None or not self.train_config.cache_dir:
            return None
        return os.path.join(self.train_config.cache_dir, "vision_encodings", f"{self.__vision_encoding_disk_key(canonical_paths, token_count)}.pt")

    def scale_latents(self, latents: Tensor) -> Tensor:
        # the FLUX.2 VAE emits 32-channel latents at 1/8 image scale, but the transformer
        # consumes a 2x2-packed 128-channel latent at 1/16 scale, normalized with the VAE's
        # batch-norm running statistics (its bn has 128 channels - one per packed channel).
        # Pack the 2x2 spatial blocks into 4x channels, then normalize in packed space.
        b, c, h, w = latents.shape
        packed = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        bn = self.vae.bn
        eps = self.vae.config.batch_norm_eps
        mean = bn.running_mean.to(device=packed.device, dtype=packed.dtype).view(1, -1, 1, 1)
        std = (bn.running_var + eps).sqrt().to(device=packed.device, dtype=packed.dtype).view(1, -1, 1, 1)
        return (packed - mean) / std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        # reverse of scale_latents: denormalize in packed 128-channel space, then unpack
        # the 2x2 blocks back to 32 channels at 2x spatial resolution for the VAE decoder.
        b, c, h, w = latents.shape
        bn = self.vae.bn
        eps = self.vae.config.batch_norm_eps
        mean = bn.running_mean.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
        std = (bn.running_var + eps).sqrt().to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
        denormed = latents * std + mean
        return denormed.view(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)

    def all_embeddings(self) -> list[MossModelEmbedding]:
        return self.additional_embeddings \
               + ([self.embedding] if self.embedding is not None else [])

    def calculate_timestep_shift(self, latent_width: int, latent_height: int):
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift

        # the MOSS transformer consumes the VAE latents at full spatial resolution
        # (VAE scale factor 16, 128 channels), so the image sequence length is the
        # latent grid size as-is.
        image_seq_len = latent_width * latent_height
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return math.exp(mu)
