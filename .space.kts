job("Package") {

    requirements {
        workerTags("swarm-worker")
    }

    container(displayName = "Build and Test", image = "python:3.11") {

        env["PYPI_USER_TOKEN"] = "{{ project:PYPI_USER_TOKEN }}"
        env["PYPI_PASSWORD_TOKEN"] = "{{ project:PYPI_PASSWORD_TOKEN }}"

        shellScript {
            content = """
                echo "hello world"
            """
            interpreter = "/bin/bash"
        }
    }
}