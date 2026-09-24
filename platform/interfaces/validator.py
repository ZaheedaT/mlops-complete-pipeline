from abc import ABC, abstractmethod
from typing import Any

from platform.core.project import ProjectConfig
from platform.core.result import Result


class ModelValidator(ABC):
    """
    Abstract interface for model validation implementations.
    """

    @abstractmethod
    def validate(
        self,
        project: ProjectConfig,
        model: Any,
        data: Any,
    ) -> Result:
        """
        Validate a trained machine learning model.
        """
        raise NotImplementedError