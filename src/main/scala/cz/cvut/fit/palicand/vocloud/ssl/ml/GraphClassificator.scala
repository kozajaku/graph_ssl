package cz.cvut.fit.palicand.vocloud.ssl.ml

import org.apache.spark.ml.classification.{ClassificationModel, Classifier}
import org.apache.spark.ml.param.{ParamValidators, IntParam, Param, Params}

/**
  * Created by palickaa on 09/03/16.
  */

trait GraphParams extends Params {
  final val neighbourhoodKernel = new Param[String](this,"neighKernel", "Kernel to use for getting neighbourhors",
                                                    ParamValidators.inArray(Array("knn", "gaussian")))
  setDefault(neighbourhoodKernel -> "knn")

  final val kNeighbours = new IntParam(this, "kNeighbours", "K in KNN", ParamValidators.gtEq(1))
  setDefault(kNeighbours -> 3)

  final val maxIterations = new IntParam(this, "maxIterations", "How long should we iterate")
  setDefault(maxIterations -> 100)
}

abstract class GraphClassificator[FeaturesType,
                         E <: Classifier[FeaturesType, E, M],
                         M <: ClassificationModel[FeaturesType, M]](override val uid: String)
  extends Classifier[FeaturesType, E, M] with GraphParams {


}
