package cz.cvut.fit.palicand.vocloud.ssl

/**
  * Created by palickaa on 08/03/16.
  */

import java.io.File
import cz.cvut.fit.palicand.vocloud.ssl.ml.classification.{GraphClassifier, LabelSpreadingClassifier, LabelPropagationClassifier}
import cz.cvut.fit.palicand.vocloud.ssl.utils.DataframeUtils
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.{VectorUDT, Vectors, Vector}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{Logging, SparkConf, SparkContext}
import org.json4s._
import org.json4s.native.JsonMethods._

import scala.collection.mutable.ArrayBuffer

case class CmdArgs(configFile : File = new File(".")) {

}

/**
  *
  * @param inputData URI to input data. May be either a path to a local system, or an HDFS URI
  * @param outputData URI to output data. Same value format as for the inputData param.
  * @param neighbourhoodKernel Kernel to use for computing neighbours. May be either KNN or Gaussian.
  * @param kernelParameters Parameters of the kernel. For details, see the documentation for the kernel.
  *
  */
case class Config(inputData: String, outputData: String, neighbourhoodKernel: String, kernelParameters: KnnKernelParameters,
                  method: String, iterations: Int, partitions: Int, methodParameters: Option[LabelSpreadParameters],
                  ratio: Double)

case class KnnKernelParameters(k: Int, bufferSize: Double, topTreeSize: Int, topTreeLeafSize: Int,
                               subTreeLeafSize: Int)

case class LabelSpreadParameters(alpha: Double)

object GraphSSL extends Logging {
  implicit val formats = DefaultFormats

  def runSparkJob(config: CmdArgs): Unit = {
    val conf = new SparkConf().setAppName("Graph-SSL").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    val jsonConfig = parse(file2JsonInput(config.configFile)).extract[Config]
    val sqlContext = new SQLContext(sc)

    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "false") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load(jsonConfig.inputData).repartition(jsonConfig.partitions)
    val rowSchema = StructType(StructField("rowNo", LongType) :: StructField("spectrum", StringType) ::
      StructField("features", new VectorUDT()) :: StructField("label", DoubleType) ::  Nil)
    /*transform the data, so that a -1 label (no label) is represented as a 6th label. This will come in handy
    in one-hot encoding and representing the label distributions as a vector
    */
    val transformed = sqlContext.createDataFrame(df.rdd.zipWithIndex.map { case (row, i) =>
      val buffer = ArrayBuffer[Double]()
      for (j <- 1 to row.length - 2) {
        buffer.append(row.getDouble(j))
      }
      Row.fromSeq(i :: row.getString(0) :: Vectors.dense(buffer.toArray) :: row.getInt(row.length - 1).toDouble :: Nil)
    }, rowSchema).filter(s"label != ${4}") //this shouuuld be more general... we are filtering out class 4 since it is a chaotic class and not really one to bring many information
    //sample the data
    val sampled = sqlContext.createDataFrame(transformed.where("label != -1").unionAll(transformed.where("label = -1").sample(withReplacement = false,
      jsonConfig.ratio)).rdd.zipWithIndex.map {case (row@Row(_, name, features, label), i) => Row.fromSeq(i :: name :: features :: label :: Nil)}, rowSchema).orderBy("rowNo")
    val numberOfLabels = DataframeUtils.numberOfClasses(sampled, "label")
    logDebug(s"${transformed.select("label").distinct().collect()}")

    val dataset = sqlContext.createDataFrame(sampled.map { case r@Row(rowNo: Long, spectra: String, vector: Vector, label: Double) =>
      if (label == -1) {
        Row.fromSeq(rowNo :: spectra :: vector :: numberOfLabels.toDouble :: Nil)
      } else {
        Row.fromSeq(rowNo :: spectra :: vector :: label :: Nil)
      }
    }, rowSchema).repartition(jsonConfig.partitions)
    logInfo(s"Size of dataset: ${dataset.count}")
    val clf : GraphClassifier = jsonConfig.method match {
      case "LabelPropagation" => new LabelPropagationClassifier()
      case "LabelSpreading" =>
        jsonConfig.methodParameters match {
          case Some(param) =>  val lsc = new LabelSpreadingClassifier()
            lsc.setAlpha(param.alpha)
            lsc
          case None => new LabelSpreadingClassifier()
        }
      case _ => throw new UnsupportedOperationException(s"Method ${jsonConfig.method} is not supported.")
    }

    clf.setLabelCol("label").setFeaturesCol("features").setKNeighbours(jsonConfig.kernelParameters.k).
      setMaxIterations(jsonConfig.iterations).setTopTreeLeafSize(jsonConfig.kernelParameters.topTreeLeafSize).
      setSubTreeLeafSize(jsonConfig.kernelParameters.subTreeLeafSize).
      setTopTreeSize(jsonConfig.kernelParameters.topTreeSize)
    clf.setBufferSize(jsonConfig.kernelParameters.bufferSize)
    val model = clf.fit(dataset)
    val names = dataset.select("rowNo", "spectrum", "features").map{ case Row(rowNo: Long, name: String, features: Vector) =>
      (rowNo, (name, features))
    }
    names.join(model.labels).sortBy(_._2._2._2, ascending = false).map {case (rowNo, ((name, features), (label, prob))) => (name, features, label, prob)}.saveAsTextFile(jsonConfig.outputData + "/labels.csv")
    names.join(model.labelWeights.toIndexedRowMatrix.rows.map {r => (r.index, r.vector.toDense.values)}).map {
      case (rowNo, (name, vector)) =>
        s"$name,${vector.mkString(",")}"
    }.saveAsTextFile(jsonConfig.outputData + "/distribution.csv")

  }


  def main(args: Array[String]) : Unit = {
    val parser = new scopt.OptionParser[CmdArgs]("Graph-SSL") {
      head("Semi-Supervised learning using label propagation", "0.1.0")
      arg[File]("config") required() valueName "<file>" action {(x, c) =>
        c.copy(configFile = x) } text "you need to input config file in json format"
      }
    parser.parse(args, CmdArgs()) match {
      case Some(config) => runSparkJob(config)


      case None =>

    }
  }
}

