import subprocess

from platform.core.exceptions import DeploymentException
from platform.core.project import ProjectConfig
from platform.core.result import Result
from platform.interfaces.deployer import ModelDeployer


class KubernetesModelDeployer(ModelDeployer):
    """
    Kubernetes implementation of model deployment.
    """

    def deploy(
        self,
        project: ProjectConfig,
        model,
    ) -> Result:

        try:
            image = model["image"]

            deployment_name = (
                f"{project.name}-deployment"
            )

            self._apply_deployment(
                project=project,
                deployment_name=deployment_name,
                image=image,
            )

            self._wait_for_rollout(
                project=project,
                deployment_name=deployment_name,
            )

            return Result.ok(
                message=(
                    f"Model '{project.model_name}' "
                    "deployed successfully."
                ),
                data={
                    "deployment": deployment_name,
                    "image": image,
                },
            )

        except subprocess.CalledProcessError as exc:
            return Result.failure(
                message=f"Kubernetes deployment failed: {exc}"
            )

        except Exception as exc:
            return Result.failure(
                message=f"Deployment failed: {exc}"
            )

    def _apply_deployment(
        self,
        project: ProjectConfig,
        deployment_name: str,
        image: str,
    ) -> None:

        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {deployment_name}
  namespace: {project.deployment_namespace}
spec:
  replicas: {project.deployment_replicas}
  selector:
    matchLabels:
      app: {deployment_name}
  template:
    metadata:
      labels:
        app: {deployment_name}
    spec:
      containers:
        - name: model
          image: {image}
          ports:
            - containerPort: 8080
"""

        process = subprocess.run(
            [
                "kubectl",
                "apply",
                "-f",
                "-",
            ],
            input=manifest,
            text=True,
            capture_output=True,
            check=True,
        )

        if process.returncode != 0:
            raise DeploymentException(
                process.stderr
            )

    def _wait_for_rollout(
        self,
        project: ProjectConfig,
        deployment_name: str,
    ) -> None:

        subprocess.run(
            [
                "kubectl",
                "rollout",
                "status",
                f"deployment/{deployment_name}",
                "-n",
                project.deployment_namespace,
                "--timeout=300s",
            ],
            check=True,
        )