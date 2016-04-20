package cz.cvut.fit.palicand.vocloud.ssl.ml

import com.holdenkarau.spark.testing._
import cz.cvut.fit.palicand.vocloud.ssl.ml.classification.LabelPropagationClassifier
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.{DenseVector, VectorUDT, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.scalatest.{Matchers, FlatSpec}

/**
  * Created by palickaa on 11/03/16.
  */
class LabelPropagationClassifierTest extends FlatSpec with SharedSparkContext with Matchers {

  behavior of "LabelPropagationTest"

  it should "train" in {
    val sqlContext = new SQLContext(sc)
    val rdd: RDD[Row] = sc.parallelize(Row(0L, 0.0, Vectors.dense(0.0, 1.0)) :: Row(1L, 1.0, Vectors.dense(1.0, 0.0)) :: Row(2L, 2.0, Vectors.dense(0.0, 0.0)) :: Nil)
    val df = sqlContext.createDataFrame(rdd, StructType(List(StructField("rowNo", LongType), StructField("label", DoubleType), StructField("features", new VectorUDT))))
    val clf = new LabelPropagationClassifier()
    clf.setKNeighbours(2)
    clf.setLabelCol("label")
    clf.setFeaturesCol("features")
    val model = clf.fit(df)
    model.labelWeights.toIndexedRowMatrix().rows.collect() should be(createIndexedRow(0, 1.0, 0.0) ::
      createIndexedRow(1, 0.0, 1.0) :: createIndexedRow(2, 1.0, 0) :: Nil)
  }

  def createIndexedRow(i: Int, vals: Double*): IndexedRow = {
    new IndexedRow(i, new DenseVector(vals.toArray))
  }
}
