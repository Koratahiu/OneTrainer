from modules.model.Krea2Model import Krea2Model
from modules.modelSaver.GenericEmbeddingModelSaver import make_embedding_model_saver
from modules.modelSaver.krea2.Krea2EmbeddingSaver import Krea2EmbeddingSaver
from modules.util.enum.ModelType import ModelType

Krea2EmbeddingModelSaver = make_embedding_model_saver(
    model_types=ModelType.KREA_2,
    model_class=Krea2Model,
    embedding_saver_class=Krea2EmbeddingSaver,
)
