from abc import ABC, abstractmethod

from mlops_platform.core.project import ProjectConfig
from mlops_platform.core.result import Result


class ModelMonitor(ABC):
    """
    Abstract interface for model monitoring implementations.
    """

    @abstractmethod
    def check(
        self,
        project: ProjectConfig,
    ) -> Result:
        """
        Check the deployed model and determine
        whether action is required.
        """
        raise NotImplementedError