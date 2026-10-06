from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectConfig:
    """
    Configuration for an MLOps project.
    """

    name: str
    model_name: str

    model_type: str
    model_framework: str
    model_algorithm: str

    training_entrypoint: Optional[str] = None

    deployment_namespace: str = "default"
    deployment_replicas: int = 1
    ecr_repository: Optional[str] = None

    monitoring_workspace: Optional[str] = None
    monitoring_project: Optional[str] = None

    @classmethod
    def from_dict(cls, config: dict) -> "ProjectConfig":
        project = config["project"]
        model = config["model"]
        training = config.get("training", {})
        deployment = config.get("deployment", {})
        monitoring = config.get("monitoring", {})

        return cls(
            name=project["name"],
            model_name=project["model_name"],

            model_type=model["type"],
            model_framework=model["framework"],
            model_algorithm=model["algorithm"],

            training_entrypoint=training.get(
                "entrypoint"
            ),

            deployment_namespace=deployment.get(
                "namespace",
                "default",
            ),

            deployment_replicas=deployment.get(
                "replicas",
                1,
            ),

            ecr_repository=deployment.get(
                "ecr_repository"
            ),

            monitoring_workspace=monitoring.get(
                "workspace"
            ),

            monitoring_project=monitoring.get(
                "project"
            ),
        )