from abc import ABC, abstractmethod
from typing import Any

from mlops_platform.core.project import ProjectConfig
from mlops_platform.core.result import Result


class DataValidator(ABC):
    """
    Abstract interface for data validation implementations.
    """

    @abstractmethod
    def validate(
        self,
        project: ProjectConfig,
        data: Any,
    ) -> Result:
        """
        Validate project data.
        """
        raise NotImplementedError