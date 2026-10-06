from abc import ABC, abstractmethod
from typing import Any

from mlops_platform.core.project import ProjectConfig
from mlops_platform.core.result import Result


class ModelRegistry(ABC):
    """
    Abstract interface for model registry implementations.
    """

    @abstractmethod
    def register(
        self,
        project: ProjectConfig,
        model: Any,
    ) -> Result:
        """
        Register a trained model.
        """
        raise NotImplementedError