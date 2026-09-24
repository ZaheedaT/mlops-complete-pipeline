from typing import Any, Dict

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from platform.core.exceptions import ValidationException
from platform.core.project import ProjectConfig
from platform.core.result import Result


class ClassificationValidator:
    """
    Validates classification models using standard
    classification metrics.
    """

    def validate(
        self,
        project: ProjectConfig,
        model: Any,
        data: Any,
    ) -> Result:

        try:
            if not isinstance(data, tuple) or len(data) != 2:
                raise ValidationException(
                    "Classification validation data must be "
                    "(X_test, y_test)."
                )

            X_test, y_test = data

            predictions = model.predict(X_test)

            metrics: Dict[str, float] = {
                "accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
                "precision": precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
            }

            return Result.ok(
                message=(
                    f"Classification model validation "
                    f"completed for '{project.name}'."
                ),
                data=metrics,
            )

        except ValidationException as exc:
            return Result.failure(
                message=str(exc)
            )

        except Exception as exc:
            return Result.failure(
                message=f"Classification validation failed: {exc}"
            )