import modules.util.multi_gpu_util as multi
from modules.util.tqdm_util import tqdm

import torch


class _BufferedBatches:
    """Yields the buffered scan batches first, then delegates to the underlying
    data loader's iterator. Both draw from the same stateful MGDS pipeline, so
    re-emitting the buffered batches and then continuing produces the original
    epoch sequence with no duplication or loss."""

    def __init__(self, buffer, data_loader):
        self.buffer = list(buffer)
        self.data_loader = data_loader

    def __iter__(self):
        yield from self.buffer
        yield from self.data_loader


class VisionInitDataLoader:
    """Wraps the training data loader to run the Moss vision-based embedding init
    as a one-shot pre-scan of the first epoch, instead of lazily inside predict().

    Embeddings created with image_init on start from a temporary text-based init
    (see BaseMossSetup._setup_embeddings, which sets image_init_pending). On the
    first epoch this wrapper iterates the stream, buffers every batch, and collects
    the image paths of the samples whose caption contains each pending embedding's
    placeholder. Once each pending embedding has image_init_count matching images
    (or the epoch is exhausted), the images are encoded through the text encoder's
    vision tower and the pooled tokens are copied in place into the registered
    append-token parameter (the optimizer holds the tensor reference, so the object
    is never replaced). The result is broadcast from rank 0 for multi-GPU
    consistency, since each rank scans its own shard.

    The buffered batches are re-emitted at the start of iteration, so no training
    data is lost to the scan: the first epoch simply front-loads the init.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.buffer = []
        self.scanned = False

    def get_data_set(self):
        return self.data_loader.get_data_set()

    def get_data_loader(self):
        if self.buffer:
            return _BufferedBatches(self.buffer, self.data_loader.get_data_loader())
        return self.data_loader.get_data_loader()

    def run_vision_init_scan(self, model, config, train_device):
        """Consume matching batches from the current epoch into the buffer and
        perform the vision init. No-op unless the model has pending image-init
        embeddings (i.e. it is a Moss model with image_init on), so it is safe to
        call unconditionally from the trainer's epoch loop."""
        if self.scanned:
            return
        self.scanned = True

        pending = [
            emb for emb in getattr(model, "all_embeddings", lambda: [])()
            if getattr(emb, "image_init_pending", False)
        ]
        if not pending:
            return

        if multi.is_master():
            tqdm.write("vision init: scanning the training stream for embedding images...")

        append_tokens = model.append_tokens
        while True:
            try:
                batch = next(iter(self.data_loader.get_data_loader()))
            except StopIteration:
                break  # epoch exhausted; init with whatever was collected
            self.buffer.append(batch)

            captions = batch.get('prompt') or []
            image_paths = batch.get('image_path') or []
            for emb in pending:
                if not emb.image_init_pending:
                    continue
                emb.image_init_paths.extend(
                    image_paths[i]
                    for i, caption in enumerate(captions)
                    if i < len(image_paths) and emb.placeholder in (caption or "")
                )
                if emb.image_init_target is None or len(emb.image_init_paths) >= emb.image_init_target:
                    emb.image_init_pending = False

            # every pending embedding reached its target (or the epoch ran out below)
            if all(not emb.image_init_pending for emb in pending):
                break

        # compute each pending embedding's vision init from this rank's shard, in place
        for emb in pending:
            if not emb.image_init_paths:
                continue  # this rank's shard had no matching images; keep the text init
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
            if multi.is_master():
                tqdm.write(
                    f"vision init: replaced append tokens of embedding '{emb.placeholder}' "
                    f"with the pooled vision encoding of {len(emb.image_init_paths)} training image(s)"
                )

        # broadcast the fixed set of pending params from rank 0 so every rank ends up with the
        # same init. The set (uuids, order) is identical on all ranks because they share the
        # config, so the collective is reached in lockstep - no deadlock even when a rank's shard
        # had no matching images for some embedding (that embedding then keeps rank 0's value).
        multi.broadcast_parameters([append_tokens[emb.uuid] for emb in pending], train_device)
