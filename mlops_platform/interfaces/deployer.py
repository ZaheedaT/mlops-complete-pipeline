from abc import ABC, abstractmethod
from typing import Any

from mlops_platform.core.project import ProjectConfig
from mlops_platform.core.result import Result


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