from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

DATASET_DOI = "doi:10.7910/DVN/V71K5R"
DATAVERSE_API_URL = "https://dataverse.harvard.edu/api"
ML4ROADSAFETY_REPO_URL = "https://github.com/VirtuosoResearch/ML4RoadSafety"
ML4ROADSAFETY_PAPER_URL = (
    "https://papers.nips.cc/paper_files/paper/2023/hash/"
    "a365be0950259c9624edfb4d26eabd46-Abstract-Datasets_and_Benchmarks.html"
)

DEFAULT_STATE = "MA"
DEFAULT_MONTHS = ("2022-01", "2022-02", "2022-03")
DEFAULT_MAX_SEGMENTS = 5000
DEFAULT_SEED = 19091985

NODE_FEATURE_COLUMNS = ("tavg", "tmin", "tmax", "prcp", "wspd", "pres")
STATIC_EDGE_FEATURE_KEYS = (
    "oneway",
    "access_ramp",
    "bus_stop",
    "crossing",
    "disused",
    "elevator",
    "escape",
    "living_street",
    "motorway",
    "motorway_link",
    "primary",
    "primary_link",
    "residential",
    "rest_area",
    "road",
    "secondary",
    "secondary_link",
    "stairs",
    "tertiary",
    "tertiary_link",
    "trunk",
    "trunk_link",
    "unclassified",
    "unsurfaced",
    "length",
)


@dataclass(frozen=True, order=True)
class MonthSpec:
    year: int
    month: int

    @classmethod
    def parse(cls, value: str) -> "MonthSpec":
        text = str(value).strip()
        try:
            year_text, month_text = text.split("-", maxsplit=1)
            year = int(year_text)
            month = int(month_text)
        except Exception as exc:
            raise ValueError(f"Mes invalido {value!r}; usa formato YYYY-MM.") from exc
        if month < 1 or month > 12:
            raise ValueError(f"Mes fuera de rango en {value!r}.")
        return cls(year=year, month=month)

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def parse_months(values: tuple[str, ...] | list[str]) -> tuple[MonthSpec, ...]:
    months = tuple(MonthSpec.parse(v) for v in values)
    if len(months) < 3:
        raise ValueError("El piloto necesita al menos tres meses: train, val y test.")
    return tuple(sorted(months))

