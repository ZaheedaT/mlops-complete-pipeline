from abc import ABC, abstractmethod
from typing import Any

from mlops_platform.core.exceptions import TrainingException
from mlops_platform.core.project import ProjectConfig


class ModelFactory(ABC):
    """
    Abstract factory for creating machine learning models.
    """

    @abstractmethod
    def create(
        self,
        project: ProjectConfig,
    ) -> Any:
        """
        Create a model based on the project configuration.
        """
        raise NotImplementedError


class SklearnModelFactory(ModelFactory):
    """
    Factory for creating scikit-learn models.
    """

    def create(
        self,
        project: ProjectConfig,
    ) -> Any:

        algorithm = project.model_algorithm.lower()

        if algorithm == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            if project.model_type == "classification":
                return RandomForestClassifier()

            if project.model_type == "regression":
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor()

        if algorithm == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression()

        if algorithm == "linear_regression":
            from sklearn.linear_model import LinearRegression

            return LinearRegression()

        raise TrainingException(
            f"Unsupported scikit-learn model algorithm: "
            f"{project.model_algorithm}"
        )


class PyTorchModelFactory(ModelFactory):
    """
    Factory for creating PyTorch models.
    """

    def create(
        self,
        project: ProjectConfig,
    ) -> Any:

        raise NotImplementedError(
            f"PyTorch model algorithm "
            f"'{project.model_algorithm}' "
            "has not been implemented."
        )


class TensorFlowModelFactory(ModelFactory):
    """
    Factory for creating TensorFlow models.
    """

    def create(
        self,
        project: ProjectConfig,
    ) -> Any:

        raise NotImplementedError(
            f"TensorFlow model algorithm "
            f"'{project.model_algorithm}' "
            "has not been implemented."
        )


class ModelFactorySelector:
    """
    Selects the appropriate model factory based on
    the configured machine learning framework.
    """

    _factories = {
        "sklearn": SklearnModelFactory,
        "pytorch": PyTorchModelFactory,
        "tensorflow": TensorFlowModelFactory,
    }

    @classmethod
    def create(
        cls,
        project: ProjectConfig,
    ) -> ModelFactory:

        if not project.model_framework:
            raise TrainingException(
                "Model framework has not been configured."
            )

        framework = project.model_framework.lower()

        factory_class = cls._factories.get(framework)

        if factory_class is None:
            raise TrainingException(
                f"Unsupported model framework: {framework}"
            )

        return factory_class()