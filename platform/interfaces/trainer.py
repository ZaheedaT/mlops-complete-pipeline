from abc import ABC, abstractmethod
from typing import Any

from platform.core.project import ProjectConfig
from platform.core.result import Result


class Trainer(ABC):
    """
    Abstract interface for model training implementations.
    """

    @abstractmethod
    def train(
        self,
        project: ProjectConfig,
        data: Any,
    ) -> Result:
        """
        Train a machine learning model.
        """
        raise NotImplementedError