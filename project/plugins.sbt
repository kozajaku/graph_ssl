logLevel := Level.Warn

//resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"
//addSbtPlugin("org.spark-packages" % "sbt-spark-package" % "0.2.3")

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/releases/"
)
addSbtPlugin("net.virtual-void" % "sbt-dependency-graph" % "0.8.2")

