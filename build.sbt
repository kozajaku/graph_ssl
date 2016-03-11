import sbt.Tests.Setup

name := "graph_ssl"

version := "1.0"

scalaVersion := "2.11.7"
libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.0"
libraryDependencies += "com.databricks" % "spark-csv_2.11" % "1.4.0"
libraryDependencies += "com.datastax.spark" %% "spark-cassandra-connector" % "1.6.0-M1"
libraryDependencies += "org.json4s" %% "json4s-native" % "3.3.0"
libraryDependencies += "com.github.scopt" %% "scopt" % "3.4.0"
libraryDependencies += "org.scalactic" %% "scalactic" % "2.2.6"
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.6" % "test"
libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % "1.6.0_0.3.1"


scalacOptions ++= Seq("-unchecked", "-explaintypes", "-deprecation", "-feature", "-Xfatal-warnings")
parallelExecution in Test := false
