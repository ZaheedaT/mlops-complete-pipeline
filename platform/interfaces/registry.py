from abc import ABC, abstractmethod
from typing import Any

from platform.core.project import ProjectConfig
from platform.core.result import Result


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