from typing import Any

from platform.core.exceptions import TrainingException
from platform.core.project import ProjectConfig
from platform.core.result import Result
from platform.interfaces.trainer import Trainer
from platform.training.model_factory import ModelFactorySelector


class ModelTrainer(Trainer):
    """
    Concrete implementation of the Trainer interface.

    Creates a model using the appropriate model factory
    and trains it using the supplied data.
    """

    def train(
        self,
        project: ProjectConfig,
        data: Any,
    ) -> Result:

        try:
            if data is None:
                raise TrainingException(
                    "Training data cannot be None."
                )

            factory = ModelFactorySelector.create(project)

            model = factory.create(project)

            model = self._fit_model(
                project=project,
                model=model,
                data=data,
            )

            return Result.ok(
                message=(
                    f"Model training completed for "
                    f"project '{project.name}'."
                ),
                data=model,
            )

        except TrainingException as exc:
            return Result.failure(
                message=str(exc)
            )

        except Exception as exc:
            return Result.failure(
                message=f"Training failed: {exc}"
            )

    def _fit_model(
        self,
        project: ProjectConfig,
        model: Any,
        data: Any,
    ) -> Any:
        """
        Fit the model using the supplied training data.
        """

        framework = project.model_framework.lower()

        if framework == "sklearn":
            return self._fit_sklearn(
                model=model,
                data=data,
            )

        if framework == "pytorch":
            return self._fit_pytorch(
                model=model,
                data=data,
            )

        if framework == "tensorflow":
            return self._fit_tensorflow(
                model=model,
                data=data,
            )

        raise TrainingException(
            f"Unsupported model framework: {framework}"
        )

    @staticmethod
    def _fit_sklearn(
        model: Any,
        data: Any,
    ) -> Any:
        """
        Train a scikit-learn model.

        Expected data format:

            (X_train, y_train)
        """

        if not isinstance(data, tuple) or len(data) != 2:
            raise TrainingException(
                "scikit-learn training data must be "
                "(X_train, y_train)."
            )

        X_train, y_train = data

        model.fit(
            X_train,
            y_train,
        )

        return model

    @staticmethod
    def _fit_pytorch(
        model: Any,
        data: Any,
    ) -> Any:
        """
        Train a PyTorch model.
        """

        raise NotImplementedError(
            "PyTorch training loop has not been implemented."
        )

    @staticmethod
    def _fit_tensorflow(
        model: Any,
        data: Any,
    ) -> Any:
        """
        Train a TensorFlow model.
        """

        raise NotImplementedError(
            "TensorFlow training loop has not been implemented."
        )