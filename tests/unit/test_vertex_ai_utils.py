from pathlib import Path


def test__load_repository_from_gcs():
    """Tests the `_load_repository_from_gcs` function against a public artifact URI. But the
    function is overriden since the client needs to be anonymous temporarily, as we're testing
    against a publicly accessible artifact.

    References:
        - https://cloud.google.com/storage/docs/public-datasets/era5
        - https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/raw/date-variable-static/2021/12/31/soil_type?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))
    """

    public_artifact_uri = (
        "gs://gcp-public-data-arco-era5/raw/date-variable-static/2021/12/31/soil_type"
    )

    def _load_repository_from_gcs(artifact_uri: str, target_dir: Path) -> str:
        """Temporarily override of the `_load_repository_from_gcs` function."""
        import re

        from google.cloud import storage
        from huggingface_inference_toolkit.vertex_ai_utils import GCS_URI_PREFIX

        if isinstance(target_dir, str):
            target_dir = Path(target_dir)

        if artifact_uri.startswith(GCS_URI_PREFIX):
            matches = re.match(f"{GCS_URI_PREFIX}(.*?)/(.*)", artifact_uri)
            bucket_name, prefix = matches.groups()  # type: ignore

            gcs_client = storage.Client.create_anonymous_client()
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

    target_dir = Path.cwd() / "target"
    target_dir_path = _load_repository_from_gcs(
        artifact_uri=public_artifact_uri, target_dir=target_dir
    )

    assert target_dir == Path(target_dir_path)
    assert Path(target_dir_path).exists()
    assert (Path(target_dir_path) / "static.nc").exists()
