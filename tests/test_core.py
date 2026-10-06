from mlops_platform.core.project import ProjectConfig
from mlops_platform.core.result import Result, PipelineResult


def test_result_success():
    result = Result.ok(
        message="Operation successful.",
        data={"value": 123},
    )

    assert result.success is True
    assert result.message == "Operation successful."
    assert result.data["value"] == 123


def test_result_failure():
    result = Result.failure(
        message="Operation failed.",
    )

    assert result.success is False
    assert result.message == "Operation failed."


def test_pipeline_result():
    result = PipelineResult(
        success=True,
        retrained=True,
        message="Pipeline completed.",
        model_id="model-123",
        deployment_successful=True,
    )

    assert result.success is True
    assert result.retrained is True
    assert result.model_id == "model-123"
    assert result.deployment_successful is True


def test_project_config_from_dict():

    config = {
        "project": {
            "name": "test-project",
            "model_name": "test-model",
        },
        "model": {
            "type": "classification",
            "framework": "sklearn",
            "algorithm": "random_forest",
        },
        "training": {
            "entrypoint": "project.train:train",
        },
        "deployment": {
            "namespace": "default",
            "replicas": 2,
            "ecr_repository": "test-project",
        },
        "monitoring": {
            "workspace": "test-workspace",
            "project": "test-project",
        },
    }

    project = ProjectConfig.from_dict(config)

    assert project.name == "test-project"
    assert project.model_name == "test-model"
    assert project.model_type == "classification"
    assert project.model_framework == "sklearn"
    assert project.model_algorithm == "random_forest"
    assert project.training_entrypoint == "project.train:train"
    assert project.deployment_namespace == "default"
    assert project.deployment_replicas == 2
    assert project.ecr_repository == "test-project"
    assert project.monitoring_workspace == "test-workspace"
    assert project.monitoring_project == "test-project"
