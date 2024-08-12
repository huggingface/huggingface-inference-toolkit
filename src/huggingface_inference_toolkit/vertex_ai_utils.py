import re
from pathlib import Path
from typing import Union

from huggingface_inference_toolkit.logging import logger

GCS_URI_PREFIX = "gs://"


# copied from https://github.com/googleapis/python-aiplatform/blob/94d838d8cfe1599bc2d706e66080c05108821986/google/cloud/aiplatform/utils/prediction_utils.py#L121
def _load_repository_from_gcs(
    artifact_uri: str, target_dir: Union[str, Path] = "/tmp"
) -> str:
    """
    Load files from GCS path to target_dir
    """
    from google.cloud import storage

    logger.info(f"Loading model artifacts from {artifact_uri} to {target_dir}")
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    if artifact_uri.startswith(GCS_URI_PREFIX):
        matches = re.match(f"{GCS_URI_PREFIX}(.*?)/(.*)", artifact_uri)
        bucket_name, prefix = matches.groups()

        gcs_client = storage.Client()
        blobs = gcs_client.list_blobs(bucket_name, prefix=prefix)
        for blob in blobs:
            name_without_prefix = blob.name[len(prefix) :]
            name_without_prefix = (
                name_without_prefix[1:]
                if name_without_prefix.startswith("/")
                else name_without_prefix
            )
            file_split = name_without_prefix.split("/")
            directory = target_dir / Path(*file_split[0:-1])
            directory.mkdir(parents=True, exist_ok=True)
            if name_without_prefix and not name_without_prefix.endswith("/"):
                blob.download_to_filename(target_dir / name_without_prefix)

    return str(target_dir.absolute())
