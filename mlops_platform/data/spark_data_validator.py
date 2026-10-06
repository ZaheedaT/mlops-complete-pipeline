from typing import Any, Dict, List

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from mlops_platform.core.exceptions import DataValidationException
from mlops_platform.core.project import ProjectConfig
from mlops_platform.core.result import Result
from mlops_platform.interfaces.data_validator import DataValidator


class SparkDataValidator(DataValidator):
    """
    Concrete data validator implementation using PySpark.

    Performs reusable data-quality checks on Spark DataFrames.
    """

    def __init__(
        self,
        required_columns: List[str] | None = None,
        allow_extra_columns: bool = True,
        max_null_ratio: float = 0.0,
        allow_duplicates: bool = False,
    ):
        self.required_columns = required_columns or []
        self.allow_extra_columns = allow_extra_columns
        self.max_null_ratio = max_null_ratio
        self.allow_duplicates = allow_duplicates

    def validate(
        self,
        project: ProjectConfig,
        data: Any,
    ) -> Result:

        try:
            if not isinstance(data, DataFrame):
                raise DataValidationException(
                    "SparkDataValidator requires a PySpark DataFrame."
                )

            checks: Dict[str, Any] = {}

            # -------------------------
            # Schema validation
            # -------------------------

            checks["schema"] = self._validate_schema(data)

            # -------------------------
            # Required columns
            # -------------------------

            checks["required_columns"] = (
                self._validate_required_columns(data)
            )

            # -------------------------
            # Null validation
            # -------------------------

            checks["null_values"] = (
                self._validate_null_values(data)
            )

            # -------------------------
            # Duplicate validation
            # -------------------------

            checks["duplicates"] = (
                self._validate_duplicates(data)
            )

            failed_checks = [
                name
                for name, result in checks.items()
                if not result["passed"]
            ]

            if failed_checks:
                return Result.failure(
                    message=(
                        f"Data validation failed for project "
                        f"'{project.name}'. "
                        f"Failed checks: {', '.join(failed_checks)}."
                    ),
                    data=checks,
                )

            return Result.ok(
                message=(
                    f"Data validation passed for project "
                    f"'{project.name}'."
                ),
                data=checks,
            )

        except DataValidationException as exc:
            return Result.failure(
                message=str(exc)
            )

        except Exception as exc:
            return Result.failure(
                message=f"Unexpected data validation error: {exc}"
            )

    def _validate_schema(
        self,
        data: DataFrame,
    ) -> Dict[str, Any]:
        """
        Verify that the input has a valid Spark schema.
        """

        schema: StructType = data.schema

        return {
            "passed": bool(schema.fields),
            "column_count": len(schema.fields),
            "columns": data.columns,
        }

    def _validate_required_columns(
        self,
        data: DataFrame,
    ) -> Dict[str, Any]:
        """
        Verify that all required columns exist.
        """

        missing_columns = [
            column
            for column in self.required_columns
            if column not in data.columns
        ]

        return {
            "passed": len(missing_columns) == 0,
            "missing_columns": missing_columns,
        }

    def _validate_null_values(
        self,
        data: DataFrame,
    ) -> Dict[str, Any]:
        """
        Check the percentage of null values in every column.
        """

        total_rows = data.count()

        if total_rows == 0:
            return {
                "passed": False,
                "reason": "Dataset contains no rows.",
            }

        null_ratios = {}

        for column in data.columns:
            null_count = data.filter(
                data[column].isNull()
            ).count()

            null_ratio = null_count / total_rows

            null_ratios[column] = null_ratio

        failed_columns = [
            column
            for column, ratio in null_ratios.items()
            if ratio > self.max_null_ratio
        ]

        return {
            "passed": len(failed_columns) == 0,
            "max_allowed_ratio": self.max_null_ratio,
            "ratios": null_ratios,
            "failed_columns": failed_columns,
        }

    def _validate_duplicates(
        self,
        data: DataFrame,
    ) -> Dict[str, Any]:
        """
        Check whether duplicate rows exist.
        """

        if self.allow_duplicates:
            return {
                "passed": True,
                "duplicates_allowed": True,
            }

        total_rows = data.count()
        distinct_rows = data.dropDuplicates().count()

        duplicate_count = total_rows - distinct_rows

        return {
            "passed": duplicate_count == 0,
            "duplicate_count": duplicate_count,
        }