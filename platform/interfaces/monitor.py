from abc import ABC, abstractmethod

from platform.core.project import ProjectConfig
from platform.core.result import Result


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