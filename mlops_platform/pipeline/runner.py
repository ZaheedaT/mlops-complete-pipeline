from pathlib import Path

from mlops_platform.config.loader import ConfigLoader
from mlops_platform.core.project import ProjectConfig

from mlops_platform.data.spark_data_validator import (
    SparkDataValidator,
)

from mlops_platform.training.model_trainer import (
    ModelTrainer,
)

from mlops_platform.validation.model_validator import (
    ModelValidationEngine,
)

from mlops_platform.registry.model_registry import (
    MLflowModelRegistry,
)

from mlops_platform.deployment.model_deployer import (
    KubernetesModelDeployer,
)

from mlops_platform.pipeline.pipeline import (
    MLOpsPipeline,
)


def load_project_config() -> ProjectConfig:
    """
    Load project configuration.
    """

    config_path = Path(
        "project/config.yaml"
    )

    loader = ConfigLoader(
        str(config_path)
    )

    config = loader.load()

    return ProjectConfig.from_dict(
        config
    )


def create_pipeline(
    project: ProjectConfig,
) -> MLOpsPipeline:

    raise NotImplementedError(
        "Pipeline dependency configuration "
        "will be implemented after all "
        "platform components are configured."
    )


def run() -> None:

    project = load_project_config()

    pipeline = create_pipeline(
        project
    )

    result = pipeline.run(
        project,
        data=None,
    )

    print(result.message)


if __name__ == "__main__":
    run()