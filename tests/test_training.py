from mlops_platform.core.project import ProjectConfig
from mlops_platform.training.model_factory import ModelFactorySelector


def create_project(
    model_type: str,
    algorithm: str,
) -> ProjectConfig:

    return ProjectConfig(
        name="test-project",
        model_name="test-model",
        model_type=model_type,
        model_framework="sklearn",
        model_algorithm=algorithm,
    )


def test_random_forest_classifier_factory():

    project = create_project(
        model_type="classification",
        algorithm="random_forest",
    )

    factory = ModelFactorySelector.create(project)
    model = factory.create(project)

    assert model.__class__.__name__ == "RandomForestClassifier"


def test_random_forest_regressor_factory():

    project = create_project(
        model_type="regression",
        algorithm="random_forest",
    )

    factory = ModelFactorySelector.create(project)
    model = factory.create(project)

    assert model.__class__.__name__ == "RandomForestRegressor"


def test_logistic_regression_factory():

    project = create_project(
        model_type="classification",
        algorithm="logistic_regression",
    )

    factory = ModelFactorySelector.create(project)
    model = factory.create(project)

    assert model.__class__.__name__ == "LogisticRegression"


def test_linear_regression_factory():

    project = create_project(
        model_type="regression",
        algorithm="linear_regression",
    )

    factory = ModelFactorySelector.create(project)
    model = factory.create(project)

    assert model.__class__.__name__ == "LinearRegression"
