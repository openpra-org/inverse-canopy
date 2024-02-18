job("Package") {

    container(displayName = "Build and Test", image = "python:3.11") {

        env["PYPI_USER_TOKEN"] = "{{ project:PYPI_USER_TOKEN }}"
        env["PYPI_PASSWORD_TOKEN"] = "{{ project:PYPI_PASSWORD_TOKEN }}"

        requirements {
            workerTags("swarm-worker")
        }

        shellScript {
            content = """
                set -euxo pipefail
                pip install --upgrade pip
                pip install twine wheel setuptools
                pip install -e .[dev]  # Install package with dev dependencies

                # Run tests without coverage
                pytest

                # Run tests with coverage
                pytest --cov=inverse_canopy --cov-report=term-missing

                # Linting
                pylint inverse_canopy

                # Build the package
                python setup.py sdist bdist_wheel

                # Check the package
                twine check dist/*

                # Push the package to PyPI
                twine upload dist/* -u ${'$'}PYPI_USER_TOKEN -p ${'$'}PYPI_PASSWORD_TOKEN
            """
            interpreter = "/bin/bash"
        }
    }
}