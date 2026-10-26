import os

import modules.util.multi_gpu_util as multi
from modules.util import path_util
from modules.util.tqdm_util import tqdm

import torch

_IMAGE_EXTENSIONS = path_util.supported_image_extensions()


def _matching_image_paths_from_disk(concepts, placeholder, target):
    """Collect up to `target` image paths whose caption contains `placeholder`,
    straight from the concept metadata and the caption files on disk.

    Mirrors the training stream's prompt selection (DataLoaderText2ImageMixin):
      - prompt_source 'sample':   per-image caption in <image>.txt (first line)
      - prompt_source 'concept':  shared caption file concept.text.prompt_path
      - prompt_source 'filename': the bare filename (no extension)

    Only file paths and small caption text are read - no images, tensors, VAE
    or text-encoder work, so this costs essentially nothing in RAM/VRAM."""
    matches = []
    for concept in concepts:
        if not concept.get('enabled', True):
            continue
        source = (concept.get('text') or {}).get('prompt_source', 'sample')
        if source == 'filename':
            continue
        path = concept.get('path', '')
        if not path or not os.path.isdir(path):
            continue
        if source == 'concept':
            caption = _read_first_line((concept.get('text') or {}).get('prompt_path', ''))
            if placeholder in caption:
                matches.extend(_list_image_files(path, concept.get('include_subdirectories', False)))
        else:
            for file_name in _list_image_files(path, concept.get('include_subdirectories', False)):
                caption = _read_first_line(file_name + '.txt')
                if placeholder in caption:
                    matches.append(file_name)
        if target is not None and len(matches) >= target:
            break
    return matches


def _read_first_line(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.readline().strip()
    except (OSError, UnicodeDecodeError):
        return ''


def _list_image_files(directory, include_subdirectories):
    files = [
        os.path.join(directory, name)
        for name in sorted(os.listdir(directory))
        if os.path.isfile(os.path.join(directory, name))
        and os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS
        and not os.path.splitext(name)[0].endswith(('-masklabel', '-condlabel'))
    ]
    if include_subdirectories:
        subdirs = [
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if os.path.isdir(os.path.join(directory, name)) and not name.startswith('.')
        ]
        for sub in subdirs:
            files.extend(_list_image_files(sub, True))
    return files


class VisionInitDataLoader:
    """Wraps the training data loader to run the Moss vision-based embedding init
    before the first epoch, without consuming (or buffering) the training stream.

    Embeddings created with image_init on start from a temporary text-based init
    (see BaseMossSetup._setup_embeddings, which sets image_init_pending). On the
    first epoch this wrapper derives, for each pending embedding, the paths of
    the training images whose caption contains its placeholder - directly from
    the concept metadata and the caption files on disk (no images, tensors, VAE
    or text-encoder work, so no meaningful RAM/VRAM cost). The images are then
    encoded through the text encoder's vision tower and the pooled tokens are
    copied in place into the registered append-token parameter (the optimizer
    holds the tensor reference, so the object is never replaced).

    The disk scan and encoding run on rank 0 only; the result is broadcast to
    all ranks, so no rank has to touch the dataset or the vision tower.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.scanned = False

    def get_data_set(self):
        return self.data_loader.get_data_set()

    def get_data_loader(self):
        return self.data_loader.get_data_loader()

    def run_vision_init_scan(self, model, config, train_device):
        """Perform the vision init from the on-disk captions. No-op unless the
        model has pending image-init embeddings (i.e. it is a Moss model with
        image_init on), so it is safe to call unconditionally from the trainer's
        epoch loop."""
        if self.scanned:
            return
        self.scanned = True

        pending = [
            emb for emb in getattr(model, "all_embeddings", lambda: [])()
            if getattr(emb, "image_init_pending", False)
        ]
        if not pending:
            return

        append_tokens = model.append_tokens

        # rank 0 scans the dataset and computes the vision init; the other ranks
        # only receive the broadcast below.
        if multi.is_master():
            tqdm.write("vision init: collecting embedding images from the dataset captions...")
            concepts = self._get_concepts()
            for emb in pending:
                if not emb.image_init_paths:
                    emb.image_init_paths = _matching_image_paths_from_disk(
                        concepts, emb.placeholder, emb.image_init_target
                    )
                if not emb.image_init_paths:
                    continue  # no matching images; keep the text init
                tokens = model.encode_images_vision_pooled_cached(
                    image_paths=emb.image_init_paths,
                    token_count=emb.token_count,
                    train_device=train_device,
                    train_dtype=model.train_dtype.torch_dtype(),
                )
                if tokens is None:
                    continue
                param = append_tokens[emb.uuid]
                with torch.no_grad():
                    param.copy_(tokens.to(device=param.device, dtype=param.dtype))
                tqdm.write(
                    f"vision init: replaced append tokens of embedding '{emb.placeholder}' "
                    f"with the pooled vision encoding of {len(emb.image_init_paths)} training image(s)"
                )

        # broadcast the fixed set of pending params from rank 0 so every rank ends up
        # with the same init. The set (uuids, order) is identical on all ranks because
        # they share the config, so the collective is reached in lockstep - no deadlock
        # even when rank 0's scan found no matching images for some embedding (that
        # embedding then keeps its text init).
        multi.broadcast_parameters([append_tokens[emb.uuid] for emb in pending], train_device)

    def _get_concepts(self) -> list:
        pipeline = self.data_loader.get_data_set().loading_pipeline
        for module in pipeline.modules:
            if hasattr(module, 'concepts') and isinstance(getattr(module, 'concepts'), list):
                return module.concepts
        return []
