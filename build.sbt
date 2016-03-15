import sbt.Tests.Setup

name := "GraphSSL"
version := "0.1.0"

crossScalaVersions := Seq("2.10.6", "2.11.8")
//sparkVersion := "1.6.1"
//sparkComponents ++= Seq("core", "mllib", "sql")
//spDependencies += "databricks/spark-csv:1.4.0-s_2.10"
libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.1" withSources()
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.1" withSources()
libraryDependencies += "com.databricks" %% "spark-csv" % "1.4.0"
libraryDependencies += "org.json4s" %% "json4s-native" % "3.3.0"
libraryDependencies += "com.github.saurfang" %% "spark-knn" % "0.2.2"
libraryDependencies += "com.github.scopt" %% "scopt" % "3.4.0"
libraryDependencies += "org.scalactic" %% "scalactic" % "2.2.6"
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.6" % "test"
libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % "1.6.0_0.3.1" excludeAll(
  ExclusionRule(organization = "org.scalatest"))
libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"
scalacOptions ++= Seq("-unchecked", "-explaintypes", "-deprecation", "-feature", "-Xfatal-warnings")
parallelExecution in Test := false
assemblyMergeStrategy in assembly := {
  case PathList("javax", "servlet", xs @ _*) => MergeStrategy.last
  case PathList("javax", "activation", xs @ _*) => MergeStrategy.last
  case PathList("javax", "xml", xs @ _*) => MergeStrategy.first
  case PathList("org", "apache", xs @ _*) => MergeStrategy.last
  case PathList("com", "google", xs @ _*) => MergeStrategy.last
  case PathList("com", "esotericsoftware", xs @ _*) => MergeStrategy.last
  case PathList("com", "codahale", xs @ _*) => MergeStrategy.last
  case PathList("com", "yammer", xs @ _*) => MergeStrategy.last
  case PathList("org", "scalatools", xs @ _*) => MergeStrategy.last
  case "about.html" => MergeStrategy.rename
  case "META-INF/ECLIPSEF.RSA" => MergeStrategy.last
  case "META-INF/mailcap" => MergeStrategy.last
  case "META-INF/mimetypes.default" => MergeStrategy.last
  case "plugin.properties" => MergeStrategy.last
  case "log4j.properties" => MergeStrategy.last
  case "index.html" => MergeStrategy.last
  case x => val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}

ivyScala := ivyScala.value map { _.copy(overrideScalaVersion = true) }
test in assembly := {}