from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Mapping

import httpx

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ml4roadsafety_validation.config import (  # noqa: E402
        DATA_DIR,
        DATASET_DOI,
        DATAVERSE_API_URL,
        DEFAULT_STATE,
    )
else:
    from .config import DATA_DIR, DATASET_DOI, DATAVERSE_API_URL, DEFAULT_STATE


REQUIRED_STATE_PATHS = (
    "adj_matrix.pt",
    "accidents_monthly.csv",
    "Edges/edge_features.pt",
    "Nodes",
)


def state_dir(data_dir: Path, state: str) -> Path:
    return Path(data_dir) / state.upper()


def normalise_extracted_layout(data_dir: Path, state: str) -> Path:
    data_dir = Path(data_dir)
    state = state.upper()
    target = state_dir(data_dir, state)
    if target.exists():
        return target

    candidates = (
        data_dir / f"ML4RoadSafety_graphs_{state}" / state,
        data_dir / f"ML4RoadSafety_graphs_{state.upper()}" / state.upper(),
    )
    for candidate in candidates:
        if candidate.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate), str(target))
            return target
    return target


def validate_state_layout(data_dir: Path, state: str = DEFAULT_STATE) -> Path:
    path = normalise_extracted_layout(Path(data_dir), state)
    missing = [rel for rel in REQUIRED_STATE_PATHS if not (path / rel).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"No se encontro un layout ML4RoadSafety valido para {state} en {path}. "
            f"Faltan: {missing_text}."
        )
    return path


def _safe_extract(zip_path: Path, extract_dir: Path) -> None:
    extract_root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (extract_root / member.filename).resolve()
            try:
                target.relative_to(extract_root)
            except ValueError as exc:
                raise ValueError(f"Entrada zip insegura: {member.filename}") from exc
        archive.extractall(extract_root)


def _dataset_metadata() -> Mapping[str, object]:
    url = f"{DATAVERSE_API_URL}/datasets/:persistentId/"
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        response = client.get(url, params={"persistentId": DATASET_DOI})
        response.raise_for_status()
        return response.json()


def _find_state_file_id(metadata: Mapping[str, object], state: str) -> int:
    expected = f"{state.upper()}.zip"
    try:
        files = metadata["data"]["latestVersion"]["files"]  # type: ignore[index]
    except Exception as exc:
        raise RuntimeError("La respuesta de Dataverse no contiene lista de archivos.") from exc
    for file_info in files:
        data_file = file_info.get("dataFile", {})
        if data_file.get("filename") == expected:
            return int(data_file["id"])
    raise FileNotFoundError(f"Dataverse no lista el archivo {expected}.")


def download_state(
    *,
    data_dir: Path = DATA_DIR,
    state: str = DEFAULT_STATE,
    skip_download: bool = False,
) -> Path:
    data_dir = Path(data_dir)
    state = state.upper()
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        return validate_state_layout(data_dir, state)
    except FileNotFoundError:
        if skip_download:
            raise

    metadata = _dataset_metadata()
    file_id = _find_state_file_id(metadata, state)
    zip_path = data_dir / f"{state}.zip"
    url = f"{DATAVERSE_API_URL}/access/datafile/{file_id}"
    timeout = httpx.Timeout(connect=60.0, read=None, write=60.0, pool=60.0)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with zip_path.open("wb") as fh:
                for chunk in response.iter_bytes():
                    if chunk:
                        fh.write(chunk)

    _safe_extract(zip_path, data_dir)
    return validate_state_layout(data_dir, state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga o valida ML4RoadSafety.")
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Solo valida un dataset ya descargado.",
    )
    args = parser.parse_args()
    path = download_state(
        data_dir=args.data_dir,
        state=args.state,
        skip_download=args.skip_download,
    )
    print(f"Dataset listo: {path}")


if __name__ == "__main__":
    main()
