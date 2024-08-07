from pathlib import Path

from huggingface_inference_toolkit.vertex_ai_utils import _load_repository_from_gcs


def test__load_repository_from_gcs():
    """Tests the `_load_repository_from_gcs` function against a public artifact URI.

    References:
        - https://cloud.google.com/storage/docs/public-datasets/era5
        - https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/raw/date-variable-static/2021/12/31/soil_type?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))
    """

    public_artifact_uri = (
        "gs://gcp-public-data-arco-era5/raw/date-variable-static/2021/12/31/soil_type"
    )
    target_dir = Path.cwd() / "target"
    target_dir_path = _load_repository_from_gcs(
        artifact_uri=public_artifact_uri, target_dir=target_dir
    )

    assert target_dir == Path(target_dir_path)
    assert Path(target_dir_path).exists()
    assert (Path(target_dir_path) / "static.nc").exists()
