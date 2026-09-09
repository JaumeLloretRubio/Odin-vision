from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FilterConfig(StrictModel):
    detect: list[str] = Field(default_factory=lambda: ["person", "car", "backpack", "drone"], max_length=64)
    ignore: list[str] = Field(default_factory=list, max_length=64)
    # None significa "usa el valor calibrado del backend activo"; el motor lo resuelve antes de guardar.
    threshold: float | None = Field(default=None, ge=0, le=1)
    margin: float | None = Field(default=None, ge=0, le=1)
    detector_interval: int = Field(default=1, ge=1, le=30)
    recognition_interval: int = Field(default=15, ge=1, le=300)
    # El VLM cuesta ~0.6 s por track nuevo en CPU: desactivarlo es la palanca para tiempo real.
    describe: bool = True
    record_labels: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("detect", "ignore", "record_labels")
    @classmethod
    def clean_lists(cls, values):
        result = list(dict.fromkeys(v.strip() for v in values))
        if any(not v or len(v) > 100 for v in result):
            raise ValueError("Las entradas deben tener entre 1 y 100 caracteres")
        return result


class Assignment(StrictModel):
    label: str = Field(min_length=1, max_length=100)
    category: str = Field(default="object", min_length=1, max_length=100)

    @field_validator("label", "category")
    @classmethod
    def strip_text(cls, value):
        if not value.strip():
            raise ValueError("El texto no puede estar vacío")
        return value.strip()


class Command(StrictModel):
    text: str = Field(min_length=1, max_length=300)
    track_id: int | None = Field(default=None, ge=1)


class Offer(StrictModel):
    type: Literal["offer"]
    sdp: str = Field(min_length=1, max_length=100000)


class Location(StrictModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
