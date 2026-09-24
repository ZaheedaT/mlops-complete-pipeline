from abc import ABC, abstractmethod
from typing import Any

from platform.core.project import ProjectConfig
from platform.core.result import Result


class ModelDeployer(ABC):
    """
    Abstract interface for model deployment implementations.
    """

    @abstractmethod
    def deploy(
        self,
        project: ProjectConfig,
        model: Any,
    ) -> Result:
        """
        Deploy a model.
        """
        raise NotImplementedError