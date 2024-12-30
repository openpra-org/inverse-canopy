job("inverse-canopy") {

    requirements {
        workerTags("swarm-worker")
    }

    val registry = "packages-space.openpra.org/p/openpra/containers/"
    val image = "inverse-canopy"
    val remote = "$registry$image"

  host("Image Tags") {
    // use kotlinScript blocks for usage of parameters
    kotlinScript("Generate slugs") { api ->

      api.parameters["commitRef"] = api.gitRevision()
      api.parameters["gitBranch"] = api.gitBranch()

      val branchName = api.gitBranch()
        .removePrefix("refs/heads/")
        .replace(Regex("[^A-Za-z0-9-]"), "-") // Replace all non-alphanumeric characters except hyphens with hyphens
        .replace(Regex("-+"), "-") // Replace multiple consecutive hyphens with a single hyphen
        .lowercase() // Convert to lower case for consistency

      val maxSlugLength = if (branchName.length > 48) 48 else branchName.length
      var branchSlug = branchName.subSequence(0, maxSlugLength).toString()
      api.parameters["branchSlug"] = branchSlug

      api.parameters["isMainBranch"] = (api.gitBranch() == "refs/heads/main").toString()

    }
  }

    host("build-image") {
      shellScript {
        interpreter = "/bin/bash"
        content = """
                        docker pull $remote:{{ branchSlug }} || true
                        docker build --tag="$remote:{{ branchSlug }}" --tag="$remote:ci-{{ run:number }}-{{ branchSlug }}" .
                        docker push "$remote:ci-{{ run:number }}-{{ branchSlug }}"
                        """
      }
    }

    parallel {

        host("Tests") {
            shellScript("pytest") {
                interpreter = "/bin/bash"
                content = """
                          docker run --rm "$remote:ci-{{ run:number }}-{{ branchSlug }}" pytest -n 4
                          """
            }
        }

        host("Coverage") {
            shellScript("pytest --cov") {
                interpreter = "/bin/bash"
                content = """
                          docker run --rm "$remote:ci-{{ run:number }}-{{ branchSlug }}" pytest --cov -n 4
                          """
            }
        }

        host("Lint & Format") {
            shellScript("ruff check") {
                interpreter = "/bin/bash"
                content = """
                          docker run --rm "$remote:ci-{{ run:number }}-{{ branchSlug }}" ruff check
                          """
            }
        }
    }

    host("Publish") {

      runIf("{{ isMainBranch }}")

      env["USER"] = "{{ project:PYPI_USER_TOKEN }}"
      env["PASSWORD"] = "{{ project:PYPI_PASSWORD_TOKEN }}"

      shellScript("build & package") {
            interpreter = "/bin/bash"
            content = """
                      docker run --rm "$remote:ci-{{ run:number }}-{{ branchSlug }}" /bin/bash -c "python -m build && twine upload dist/* -u ${'$'}USER -p ${'$'}PASSWORD"
                      """
      }
    }
}