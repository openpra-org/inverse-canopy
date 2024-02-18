job("Package") {

    requirements {
        workerTags("swarm-worker")
    }

    container(displayName = "Show work dir", image = "ubuntu") {
        shellScript {
            interpreter = "/bin/bash"
            // note that you should escape the $ symbol in a Kotlin way
            content = """
                echo The working directory is
            """
        }
    }

}