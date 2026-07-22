from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# CARGA DEL MODELO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "modelo_inasistencia.joblib"
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo en: {MODEL_PATH}"
    )


model_package: dict[str, Any] = joblib.load(
    MODEL_PATH
)

model = model_package["model"]

feature_columns: list[str] = model_package[
    "feature_columns"
]


# ============================================================
# APLICACIÓN FASTAPI
# ============================================================

app = FastAPI(
    title="API de predicción de inasistencia",
    description=(
        "Predice la probabilidad de que un paciente "
        "no asista a una cita nutricional."
    ),
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://rubi-ramos.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELO DE ENTRADA
# ============================================================

class AppointmentPredictionInput(BaseModel):
    age: int = Field(
        ...,
        ge=0,
        le=120,
        description=(
            "Edad del paciente en la fecha de la cita."
        ),
        examples=[35],
    )

    appointment_month: int = Field(
        ...,
        ge=1,
        le=12,
        description=(
            "Mes de la cita, de 1 a 12."
        ),
        examples=[7],
    )

    appointment_day_of_month: int = Field(
        ...,
        ge=1,
        le=31,
        description=(
            "Día del mes de la cita."
        ),
        examples=[15],
    )

    previous_completed_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Porcentaje de citas anteriores completadas."
        ),
        examples=[75.0],
    )

    previous_no_show_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Porcentaje de citas anteriores "
            "a las que el paciente no asistió."
        ),
        examples=[25.0],
    )

    previous_cancelled_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Porcentaje de citas anteriores canceladas."
        ),
        examples=[0.0],
    )


# ============================================================
# MODELO DE RESPUESTA
# ============================================================

class AppointmentPredictionOutput(BaseModel):
    no_show_probability: float

    attendance_probability: float

    no_show_percentage: float

    attendance_percentage: float

    risk_level: Literal[
        "Bajo",
        "Medio",
        "Alto",
    ]

    predicted_no_show: int

    prediction: str

    target_name: str

    model_name: str


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def validate_historical_percentages(
    completed_percentage: float,
    no_show_percentage: float,
    cancelled_percentage: float,
) -> None:
    total_percentage = (
        completed_percentage
        + no_show_percentage
        + cancelled_percentage
    )

    no_previous_appointments = (
        completed_percentage == 0
        and no_show_percentage == 0
        and cancelled_percentage == 0
    )

    valid_total = (
        abs(
            total_percentage - 100
        ) <= 0.05
    )

    if not (
        no_previous_appointments
        or valid_total
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Los porcentajes históricos deben sumar "
                "aproximadamente 100, o los tres deben ser "
                "0 cuando el paciente no tiene citas previas."
            ),
        )


# ============================================================
# RUTA PRINCIPAL
# ============================================================

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service":
            "Predicción de inasistencia",

        "status":
            "online",

        "model":
            model_package.get(
                "model_name",
                "modelo_desconocido",
            ),

        "target":
            model_package.get(
                "target_column",
                "inasistencia_cita",
            ),

        "version":
            "2.0.0",
    }


# ============================================================
# ESTADO DEL SERVICIO
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "success":
            True,

        "model_loaded":
            True,

        "model_name":
            model_package.get(
                "model_name",
                "modelo_desconocido",
            ),

        "target_column":
            model_package.get(
                "target_column",
                "inasistencia_cita",
            ),

        "feature_columns":
            feature_columns,

        "metrics":
            model_package.get(
                "metrics",
                {},
            ),

        "trained_at_utc":
            model_package.get(
                "trained_at_utc",
            ),
    }


# ============================================================
# PREDICCIÓN
# ============================================================

@app.post(
    "/predict",
    response_model=AppointmentPredictionOutput,
)
def predict_attendance(
    input_data: AppointmentPredictionInput,
) -> AppointmentPredictionOutput:
    try:
        validate_historical_percentages(
            input_data
                .previous_completed_percentage,

            input_data
                .previous_no_show_percentage,

            input_data
                .previous_cancelled_percentage,
        )

        input_values = (
            input_data.model_dump()
        )

        features = pd.DataFrame(
            [
                {
                    column:
                        input_values[column]

                    for column in
                        feature_columns
                }
            ],
            columns=feature_columns,
        )

        probabilities = model.predict_proba(
            features
        )[0]

        model_classes = list(
            model.classes_
        )

        if 1 not in model_classes:
            raise RuntimeError(
                "El modelo no contiene la clase "
                "positiva de inasistencia."
            )

        inasistencia_class_index = (
            model_classes.index(1)
        )

        inasistencia_probability = float(
            probabilities[
                inasistencia_class_index
            ]
        )

        inasistencia_probability = min(
            max(
                inasistencia_probability,
                0.0,
            ),
            1.0,
        )

        attendance_probability = (
            1.0
            - inasistencia_probability
        )

        predicted_inasistencia = int(
            inasistencia_probability >= 0.50
        )

        risk_level: Literal[
            "Bajo",
            "Medio",
            "Alto",
        ]

        if inasistencia_probability >= 0.50:
            risk_level = "Alto"

        elif inasistencia_probability >= 0.25:
            risk_level = "Medio"

        else:
            risk_level = "Bajo"

        prediction = (
            "No asistirá"
            if predicted_inasistencia == 1
            else "Sí asistirá"
        )

        return AppointmentPredictionOutput(
            no_show_probability=round(
                inasistencia_probability,
                6,
            ),

            attendance_probability=round(
                attendance_probability,
                6,
            ),

            no_show_percentage=round(
                inasistencia_probability * 100,
                2,
            ),

            attendance_percentage=round(
                attendance_probability * 100,
                2,
            ),

            risk_level=risk_level,

            predicted_no_show=(
                predicted_inasistencia
            ),

            prediction=prediction,

            target_name=(
                "inasistencia_cita"
            ),

            model_name=model_package.get(
                "model_name",
                "modelo_desconocido",
            ),
        )

    except HTTPException:
        raise

    except KeyError as error:
        raise HTTPException(
            status_code=400,
            detail=(
                "Falta una variable requerida "
                f"por el modelo: {error}"
            ),
        ) from error

    except Exception as error:
        print(
            "Error al generar la predicción:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "No se pudo generar la predicción "
                "de inasistencia."
            ),
        ) from error