from modules.model.Krea2Model import Krea2Model
from modules.modelLoader.GenericEmbeddingModelLoader import make_embedding_model_loader
from modules.modelLoader.krea2.Krea2EmbeddingLoader import Krea2EmbeddingLoader
from modules.modelLoader.krea2.Krea2ModelLoader import Krea2ModelLoader
from modules.util.enum.ModelType import ModelType

Krea2EmbeddingModelLoader = make_embedding_model_loader(
    model_spec_map={ModelType.KREA_2: "resources/sd_model_spec/krea2-embedding.json"},
    model_class=Krea2Model,
    model_loader_class=Krea2ModelLoader,
    embedding_loader_class=Krea2EmbeddingLoader,
)
