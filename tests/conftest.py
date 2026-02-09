from pytest import fixture


@fixture(scope="session")
def example_fixture():
    return {"key": "value"}
