from typing import Any

import mlflow
from mlflow import MlflowClient

from platform.core.exceptions import RegistrationException
from platform.core.project import ProjectConfig
from platform.core.result import Result
from platform.interfaces.registry import ModelRegistry


class MLflowModelRegistry(ModelRegistry):
    """
    MLflow implementation of the model registry.
    """

    def __init__(
        self,
        tracking_uri: str,
    ):
        self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient(
            tracking_uri=tracking_uri
        )

    def register(
        self,
        project: ProjectConfig,
        model: Any,
    ) -> Result:

        try:
            if model is None:
                raise RegistrationException(
                    "Cannot register a None model."
                )

            experiment = mlflow.get_experiment_by_name(
                project.name
            )

            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    project.name
                )
            else:
                experiment_id = experiment.experiment_id

            with mlflow.start_run(
                experiment_id=experiment_id
            ) as run:

                run_id = run.info.run_id

                self._log_model(
                    project=project,
                    model=model,
                )

                model_uri = f"runs:/{run_id}/model"

                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=project.model_name,
                )

            return Result.ok(
                message=(
                    f"Model '{project.model_name}' "
                    "registered successfully."
                ),
                data={
                    "model_name": registered_model.name,
                    "version": registered_model.version,
                    "run_id": run_id,
                    "model_uri": model_uri,
                },
            )

        except RegistrationException as exc:
            return Result.failure(
                message=str(exc)
            )

        except Exception as exc:
            return Result.failure(
                message=f"Model registration failed: {exc}"
            )

    def _log_model(
        self,
        project: ProjectConfig,
        model: Any,
    ) -> None:

        framework = project.model_framework.lower()

        if framework == "sklearn":
            import mlflow.sklearn

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
            )
            return

        if framework == "pytorch":
            import mlflow.pytorch

            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
            )
            return

        if framework == "tensorflow":
            import mlflow.tensorflow

            mlflow.tensorflow.log_model(
                model,
                artifact_path="model",
            )
            return

        raise RegistrationException(
            f"Unsupported model framework: {framework}"
        )