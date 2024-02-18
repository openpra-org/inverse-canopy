job("Package") {

    requirements {
        workerTags("swarm-worker")
    }

    val registry = "packages.space.openpra.org/p/openpra/containers/"
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
    }
  }

    host("Build & Test") {

      shellScript("build"){
        interpreter = "/bin/bash"
        content = """
                          docker build --tag="$remote:{{ branchSlug }}" .
                          """
      }

      shellScript("tests"){
        interpreter = "/bin/bash"
        content = """
                          docker run --rm -it --tag="$remote:{{ branchSlug }}" pytest
                          """
      }

      shellScript("coverage"){
        interpreter = "/bin/bash"
        content = """
                          docker run --rm -it --tag="$remote:{{ branchSlug }}" pytest --cov
                          """
      }

      shellScript("lint"){
        interpreter = "/bin/bash"
        content = """
                          docker run --rm -it --tag="$remote:{{ branchSlug }}" pylint
                          """
      }
}