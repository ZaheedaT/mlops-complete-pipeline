import subprocess

import boto3

from platform.core.exceptions import ContainerRegistryException
from platform.core.project import ProjectConfig
from platform.core.result import Result


class ECRContainerRegistry:
    """
    AWS ECR implementation of the container registry.
    """

    def __init__(
        self,
        region: str,
    ):
        self.region = region

        self.ecr = boto3.client(
            "ecr",
            region_name=region,
        )

    def push(
        self,
        project: ProjectConfig,
        image: str,
    ) -> Result:

        try:
            repository = project.ecr_repository

            if not repository:
                raise ContainerRegistryException(
                    "ECR repository has not been configured."
                )

            account_id = self.ecr.get_caller_identity()
            del account_id

            response = self.ecr.describe_repositories(
                repositoryNames=[repository]
            )

            repository_uri = response[
                "repositories"
            ][0]["repositoryUri"]

            registry = repository_uri.split("/")[0]

            login_password = subprocess.check_output(
                [
                    "aws",
                    "ecr",
                    "get-login-password",
                    "--region",
                    self.region,
                ],
                text=True,
            )

            subprocess.run(
                [
                    "docker",
                    "login",
                    "--username",
                    "AWS",
                    "--password-stdin",
                    registry,
                ],
                input=login_password,
                text=True,
                check=True,
            )

            remote_image = (
                f"{repository_uri}:"
                f"{image.split(':')[-1]}"
            )

            subprocess.run(
                [
                    "docker",
                    "tag",
                    image,
                    remote_image,
                ],
                check=True,
            )

            subprocess.run(
                [
                    "docker",
                    "push",
                    remote_image,
                ],
                check=True,
            )

            return Result.ok(
                message=(
                    f"Container image pushed to "
                    f"ECR: {remote_image}"
                ),
                data={
                    "image": remote_image,
                    "repository": repository,
                },
            )

        except Exception as exc:
            return Result.failure(
                message=f"ECR push failed: {exc}"
            )