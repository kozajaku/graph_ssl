package cz.cvut.fit.palicand.vocloud.ssl.ml

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.DistributedMatrix
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.Saveable

/**
  * Created by palickaa on 09/03/16.
  */
class LabelPropagationModel(override val uid:String, val labels: DistributedMatrix) extends ClassificationModel[Vector, LabelPropagationModel] with Serializable {
  override def numClasses: Int = {
    throw new NotImplementedError()
  }

  override protected def predictRaw(features: Vector): Vector = {
    throw new NotImplementedError()
  }

  override def copy(extra: ParamMap): LabelPropagationModel = defaultCopy(extra)

  def this(labels: DistributedMatrix) = this(Identifiable.randomUID("lpc"), labels)

}
