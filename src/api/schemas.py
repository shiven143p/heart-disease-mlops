
"""Pydantic request and response schemas for the API."""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class PatientInput(BaseModel):
    """Validate patient information for heart disease prediction."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)

    trestbps: float = Field(gt=0)
    chol: float = Field(ge=0)

    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)

    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)

    oldpeak: float = Field(ge=0)
    slope: int = Field(ge=1, le=3)

    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, value: int) -> int:
        """Accept only known UCI thal categories."""

        if value not in {3, 6, 7}:
            raise ValueError(
                "thal must be one of 3, 6, or 7"
            )

        return value


class PredictionResponse(BaseModel):
    """Schema for model prediction responses."""

    prediction: int
    risk: str
    probability: float
    confidence: float
