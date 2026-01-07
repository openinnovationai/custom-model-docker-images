"""Pre-download model and tokenizer to cache."""

import os

from fairseq2.assets import AssetDownloadManager, AssetStore
from fairseq2.data.tokenizers.ref import resolve_tokenizer_reference
from fairseq2.runtime.dependency import get_dependency_resolver

MODEL_NAME = os.getenv("MODEL_NAME", "omniASR_CTC_300M_v2")


def preload_model():
    """Download model and tokenizer into cache."""
    print(f"Pre-downloading model: {MODEL_NAME}")

    resolver = get_dependency_resolver()
    asset_store = resolver.resolve(AssetStore)
    download_manager = resolver.resolve(AssetDownloadManager)

    # Retrieve model's asset card
    card = asset_store.retrieve_card(MODEL_NAME)

    # Download model checkpoint
    checkpoint_uri = card.field("checkpoint").as_uri()
    print(f"Downloading model checkpoint from: {checkpoint_uri}")
    checkpoint_path = download_manager.download_model(
        checkpoint_uri, MODEL_NAME, progress=True
    )
    print(f"Model checkpoint downloaded to: {checkpoint_path}")

    # Resolve tokenizer reference and download tokenizer file
    tokenizer_card = resolve_tokenizer_reference(asset_store, card)
    tokenizer_uri = tokenizer_card.field("tokenizer").as_uri()
    print(f"Downloading tokenizer from: {tokenizer_uri}")
    tokenizer_path = download_manager.download_tokenizer(
        tokenizer_uri, tokenizer_card.name, progress=True
    )
    print(f"Tokenizer downloaded to: {tokenizer_path}")

    print(f"All assets for {MODEL_NAME} downloaded successfully!")


if __name__ == "__main__":
    preload_model()
