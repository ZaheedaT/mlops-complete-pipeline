from typing import Any

from platform.core.exceptions import ValidationException
from platform.core.project import ProjectConfig
from platform.core.result import Result
from platform.interfaces.validator import ModelValidator
from platform.validation.classification_validator import (
    ClassificationValidator,
)
from platform.validation.regression_validator import (
    RegressionValidator,
)


class ModelValidationEngine(ModelValidator):
    """
    Selects the appropriate model validation strategy
    based on the project's model type.
    """

    def __init__(self):
        self._validators = {
            "classification": ClassificationValidator(),
            "regression": RegressionValidator(),
        }

    def validate(
        self,
        project: ProjectConfig,
        model: Any,
        data: Any,
    ) -> Result:

        try:
            if not project.model_type:
                raise ValidationException(
                    "Model type has not been configured."
                )

            model_type = project.model_type.lower()

            validator = self._validators.get(
                model_type
            )

            if validator is None:
                raise ValidationException(
                    f"Unsupported model type: {model_type}"
                )

            return validator.validate(
                project=project,
                model=model,
                data=data,
            )

        except ValidationException as exc:
            return Result.failure(
                message=str(exc)
            )

        except Exception as exc:
            return Result.failure(
                message=f"Model validation failed: {exc}"
            )