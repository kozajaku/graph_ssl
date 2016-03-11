package cz.cvut.fit.palicand.vocloud.ssl.ml

import com.holdenkarau.spark.testing._
import org.apache.avro.ipc.specific.Person
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{IntegerType, DoubleType, StructField, StructType}
import org.apache.spark.sql.{Row, SQLContext, DataFrame}
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Prop.{forAll}
import org.scalatest.FlatSpec
import org.scalatest.prop.Checkers
import org.apache.spark.mllib.linalg.{Vectors, VectorUDT, Vector}
/**
  * Created by palickaa on 11/03/16.
  */
class LabelPropagationClassifierTest extends FlatSpec with SharedSparkContext with Checkers {

  behavior of "LabelPropagationTest"

  it should "train" in {
    val sqlContext = new SQLContext(sc)
    val rdd: RDD[Row] = sc.parallelize(Row(0.0, Vectors.dense(0.0, 1.0)) :: Row(1.0, Vectors.dense(1.0, 0.0)) :: Row(2.0, Vectors.dense(0.0, 0.0)) :: Nil)
    val df = sqlContext.createDataFrame(rdd, StructType(List(StructField("label", DoubleType), StructField("features", new VectorUDT))))
    val clf = new LabelPropagationClassifier()
    clf.setLabelCol("label")
    clf.setFeaturesCol("features")
    val model = clf.fit(df)
    model.labels.numCols() == 1
  }
}
