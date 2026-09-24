class PipelineException(Exception):
    """
    Base exception for the MLOps platform.
    """

    pass


class DataValidationException(PipelineException):
    """Raised when data validation fails."""

    pass


class TrainingException(PipelineException):
    """Raised when model training fails."""

    pass


class ValidationException(PipelineException):
    """Raised when model validation fails."""

    pass


class RegistrationException(PipelineException):
    """Raised when model registration fails."""

    pass


class PackagingException(PipelineException):
    """Raised when model packaging fails."""

    pass


class ContainerRegistryException(PipelineException):
    """Raised when container registry operations fail."""

    pass


class DeploymentException(PipelineException):
    """Raised when model deployment fails."""

    pass


class MonitoringException(PipelineException):
    """Raised when model monitoring fails."""

    pass