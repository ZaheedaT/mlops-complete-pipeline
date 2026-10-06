from typing import Any, Dict

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from mlops_platform.core.exceptions import ValidationException
from mlops_platform.core.project import ProjectConfig
from mlops_platform.core.result import Result


class RegressionValidator:
    """
    Validates regression models using standard
    regression metrics.
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
                    "Regression validation data must be "
                    "(X_test, y_test)."
                )

            X_test, y_test = data

            predictions = model.predict(X_test)

            metrics: Dict[str, float] = {
                "mae": mean_absolute_error(
                    y_test,
                    predictions,
                ),
                "rmse": mean_squared_error(
                    y_test,
                    predictions,
                    squared=False,
                ),
                "r2": r2_score(
                    y_test,
                    predictions,
                ),
            }

            return Result.ok(
                message=(
                    f"Regression model validation "
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
                message=f"Regression validation failed: {exc}"
            )