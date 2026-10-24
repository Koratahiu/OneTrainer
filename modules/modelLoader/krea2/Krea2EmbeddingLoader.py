from modules.model.Krea2Model import Krea2Model
from modules.modelLoader.mixin.EmbeddingLoaderMixin import EmbeddingLoaderMixin
from modules.util.ModelNames import ModelNames


class Krea2EmbeddingLoader(
    EmbeddingLoaderMixin
):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: Krea2Model,
            directory: str,
            model_names: ModelNames,
    ):
        self._load(model, directory, model_names)
