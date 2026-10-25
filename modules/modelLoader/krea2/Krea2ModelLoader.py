import os
import traceback

from modules.model.Krea2Model import Krea2Model
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    GGUFQuantizationConfig,
    Krea2Pipeline,
    Krea2Transformer2DModel,
)
from transformers import Qwen2Tokenizer, Qwen3VLModel


class Krea2ModelLoader(
    HFModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def __load_internal(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        if os.path.isfile(os.path.join(base_model_name, "meta.json")):
            self.__load_diffusers(
                model, model_type, weight_dtypes, base_model_name, transformer_model_name, vae_model_name,
                quantization, stream_from_disk,
            )
        else:
            raise Exception("not an internal model")

    def __load_diffusers(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
            stream_from_disk: bool,
    ):
        model.tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name,
            subfolder="tokenizer",
        )

        model.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_name,
            subfolder="scheduler",
        )

        model.text_encoder, model.materialize_fn["text_encoder"] = self._load_text_encoder(
            Qwen3VLModel,
            weight_dtypes.text_encoder,
            weight_dtypes.fallback_train_dtype,
            base_model_name,
            "text_encoder",
            stream_from_disk=stream_from_disk,
        )

        model.vae = self._load_vae(
            AutoencoderKLQwenImage,
            weight_dtypes.vae,
            weight_dtypes.train_dtype,
            base_model_name,
            vae_model_name,
        )

        model.transformer, model.materialize_fn["transformer"] = self._load_transformer(
            Krea2Transformer2DModel,
            weight_dtypes,
            base_model_name,
            transformer_model_name,
            quantization,
            config=base_model_name,
            stream_from_disk=stream_from_disk,
        )

    def __load_safetensors(
            self,
            model: Krea2Model,
            model_type: ModelType,
            weight_dtypes: ModelWeightDtypes,
            base_model_name: str,
            transformer_model_name: str,
            vae_model_name: str,
            quantization: QuantizationConfig,
    ):
        transformer = Krea2Transformer2DModel.from_single_file(
            #always load transformer separately even though Krea2Pipeline.from_single_file() could load it, to avoid loading in float32:
            transformer_model_name if transformer_model_name else base_model_name,
            torch_dtype = torch.bfloat16 if weight_dtypes.transformer.torch_dtype() is None else weight_dtypes.transformer.torch_dtype(),
        )
        pipeline = Krea2Pipeline.from_single_file(
            pretrained_model_link_or_path=base_model_name,
            transformer=transformer,
        )

        if vae_model_name:
            vae = self._load_diffusers_sub_module(
                AutoencoderKLQwenImage,
                weight_dtypes.vae,
                weight_dtypes.train_dtype,
                vae_model_name,
            )
        else:
            vae = self._convert_diffusers_sub_module_to_dtype(
                pipeline.vae, weight_dtypes.vae, weight_dtypes.train_dtype
            )

        if pipeline.text_encoder is not None:
            text_encoder = self._convert_transformers_sub_module_to_dtype(
                pipeline.text_encoder, weight_dtypes.text_encoder, weight_dtypes.fallback_train_dtype
            )
            tokenizer = pipeline.tokenizer
        else:
            text_encoder = None
            tokenizer = None
            print("text encoder not loaded, continuing without it")

        transformer = self._convert_diffusers_sub_module_to_dtype(
            pipeline.transformer, weight_dtypes.transformer, weight_dtypes.train_dtype, quantization,
        )

        model.model_type = model_type
        model.tokenizer = tokenizer
        model.noise_scheduler = pipeline.scheduler
        model.text_encoder = text_encoder
        model.vae = vae
        model.transformer = transformer

    def load( #TODO share code between models
            self,
            model: Krea2Model,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
            stream_from_disk: bool = False,
    ):
        stacktraces = []

        try:
            self.__load_internal(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, quantization, stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_diffusers(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model,
                model_names.vae_model, quantization, stream_from_disk,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        try:
            self.__load_safetensors(
                model, model_type, weight_dtypes, model_names.base_model, model_names.transformer_model, model_names.vae_model, quantization,
            )
            return
        except Exception:
            stacktraces.append(traceback.format_exc())

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load model: " + model_names.base_model)
