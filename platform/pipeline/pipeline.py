from platform.core.project import ProjectConfig
from platform.core.result import PipelineResult

from platform.interfaces.trainer import Trainer
from platform.interfaces.data_validator import DataValidator
from platform.interfaces.validator import ModelValidator
from platform.interfaces.registry import ModelRegistry
from platform.interfaces.deployer import ModelDeployer
from platform.interfaces.monitor import ModelMonitor


class MLOpsPipeline:
    """
    Orchestrates the complete MLOps lifecycle.
    """

    def __init__(
        self,
        data_validator: DataValidator,
        monitor: ModelMonitor,
        trainer: Trainer,
        validator: ModelValidator,
        registry: ModelRegistry,
        deployer: ModelDeployer,
    ):
        self.data_validator = data_validator
        self.monitor = monitor
        self.trainer = trainer
        self.validator = validator
        self.registry = registry
        self.deployer = deployer

    def run(
        self,
        project: ProjectConfig,
        data,
    ) -> PipelineResult:

        data_result = self.data_validator.validate(
            project,
            data,
        )

        if not data_result.success:
            return PipelineResult(
                success=False,
                retrained=False,
                message=data_result.message,
            )

        monitor_result = self.monitor.check(
            project
        )

        if not monitor_result.success:
            return PipelineResult(
                success=False,
                retrained=False,
                message=monitor_result.message,
            )

        if not monitor_result.data:
            return PipelineResult(
                success=True,
                retrained=False,
                message="No retraining required.",
            )

        training_result = self.trainer.train(
            project,
            data,
        )

        if not training_result.success:
            return PipelineResult(
                success=False,
                retrained=True,
                message=training_result.message,
            )

        model = training_result.data

        validation_result = self.validator.validate(
            project,
            model,
            data,
        )

        if not validation_result.success:
            return PipelineResult(
                success=False,
                retrained=True,
                message=validation_result.message,
            )

        registry_result = self.registry.register(
            project,
            model,
        )

        if not registry_result.success:
            return PipelineResult(
                success=False,
                retrained=True,
                message=registry_result.message,
            )

        registered_model = registry_result.data

        deployment_result = self.deployer.deploy(
            project,
            registered_model,
        )

        if not deployment_result.success:
            return PipelineResult(
                success=False,
                retrained=True,
                model_id=project.model_name,
                deployment_successful=False,
                message=deployment_result.message,
            )

        return PipelineResult(
            success=True,
            retrained=True,
            model_id=project.model_name,
            deployment_successful=True,
            message="Pipeline completed successfully.",
        )