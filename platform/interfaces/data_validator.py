from abc import ABC, abstractmethod
from typing import Any

from platform.core.project import ProjectConfig
from platform.core.result import Result


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