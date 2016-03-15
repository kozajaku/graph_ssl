package cz.cvut.fit.palicand.vocloud.ssl

/**
  * Created by palickaa on 08/03/16.
  */

import java.io.File
import cz.cvut.fit.palicand.vocloud.ssl.ml.classification.{GraphClassifier, LabelSpreadingClassifier, LabelPropagationClassifier}
import cz.cvut.fit.palicand.vocloud.ssl.utils.DataframeUtils
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
                  method: String, methodParameters: Option[LabelSpreadParameters])

case class KnnKernelParameters(k: Int)

case class LabelSpreadParameters(alpha: Double)

object GraphSSL extends Logging {
  def main(args: Array[String]) : Unit = {
    val parser = new scopt.OptionParser[CmdArgs]("Graph-SSL") {
      head("Semi-Supervised learning using label propagation", "0.1.0")
      arg[File]("config") required() valueName "<file>" action {(x, c) =>
        c.copy(configFile = x) } text "you need to input config file in json format"
      }
    implicit val formats = DefaultFormats
    parser.parse(args, CmdArgs()) match {
      case Some(config) => {
        logDebug("test")
        val conf = new SparkConf().setAppName("Graph-SSL").setMaster("local")
        val sc = new SparkContext(conf)
        val jsonConfig = parse(file2JsonInput(config.configFile)).extract[Config]
        val sqlContext = new SQLContext(sc)
        val df = sqlContext.read
          .format("com.databricks.spark.csv")
          .option("header", "false") // Use first line of all files as header
          .option("inferSchema", "true") // Automatically infer data types
          .load(jsonConfig.inputData)
        val rowSchema = StructType(StructField("rowNo", LongType) :: StructField("spectra", StringType) ::
          StructField("features", new VectorUDT()) :: StructField("label", DoubleType) ::  Nil)
        val transformed = sqlContext.createDataFrame(df.rdd.zipWithIndex.map { case (row, i) =>
          val buffer = ArrayBuffer[Double]()
          for (j <- 1 to row.length - 2) {
            buffer.append(row.getDouble(j))
          }
          Row.fromSeq(i :: row.getString(0) :: Vectors.dense(buffer.toArray) :: row.getInt(row.length - 1).toDouble :: Nil)
        }, rowSchema)
        val numberOfLabels = DataframeUtils.numberOfClasses(transformed, "label")
        logDebug(s"${transformed.select("label").distinct().collect()}")
        val dataset = sqlContext.createDataFrame(transformed.map { case r@Row(rowNo: Long, spectra: String, vector: Vector, label: Double) =>
          if (label == -1) {
            Row.fromSeq(rowNo :: spectra :: vector :: numberOfLabels.toDouble :: Nil)
          } else {
            Row.fromSeq(rowNo :: spectra :: vector :: label :: Nil)
          }
        }, rowSchema).cache()
        logDebug(s"Size of dataset: ${dataset.count}")
        logInfo(s"Parameters: $jsonConfig")
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

        clf.setLabelCol("label").setFeaturesCol("features").setKNeighbours(jsonConfig.kernelParameters.k)
        val model = clf.fit(dataset)
        logDebug(s"Number of labels ${model.labels.size}")
        logInfo(s"Weights:\n${model.labelWeights.toLocalMatrix()}")
        logInfo(s"Labels:\n${model.labels}")


      }
      case None =>

    }
  }
}

