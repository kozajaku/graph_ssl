package cz.cvut.fit.palicand.vocloud.ssl.ml.classification

import cz.cvut.fit.palicand.vocloud.ssl.utils.{MatrixUtils, DataframeUtils}
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.classification.{RandomForestClassifier, ProbabilisticClassificationModel}
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.param.{ParamValidators, DoubleParam, Params, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg._
import org.apache.spark.sql.{Row, DataFrame}
import org.scalactic._
import Tolerance._
import TripleEquals._
import scala.annotation.tailrec

/**
  * Created by palickaa on 16/03/16.
  */


trait LabelSpreadingParams extends Params with KnnKernel {
  val alpha = new DoubleParam(this, "alpha", "The weight of label retention vs label propagation",
    ParamValidators.inRange(0.0, 1.0))

  def setAlpha(value: Double): LabelSpreadingParams = {
    set(alpha, value)
  }
}

final class LabelSpreadingClassifier(override val uid: String) extends GraphClassifier(uid)
  with LabelSpreadingParams {

  def this()  = this(Identifiable.randomUID("lsc"))


  override protected def computeLabelProbabilities(distances: BlockMatrix, toLabel: BlockMatrix, labeledRows: Long) : BlockMatrix = {
    val mulLabels = new BlockMatrix(toLabel.blocks.map {case ((i, j), mat) =>
      ((i, j), MatrixUtils.fromBreeze(MatrixUtils.toBreeze(mat) * (1.0d - $(alpha))))
    }, toLabel.rowsPerBlock, toLabel.rowsPerBlock, toLabel.numRows, toLabel.numCols).cache()
    @tailrec
    def labelSpreadingRec(laplacian: BlockMatrix, labels: BlockMatrix, iteration: Int): BlockMatrix = {
      if (iteration > $(maxIterations)) {
        return labels
      }
      val newLabels = laplacian.multiply(labels).add(mulLabels)
      //assert(MatrixUtils.hasOnlyValidElements(newLabels))
      if (hasConverged(labels, newLabels, 0.001)) {
        return newLabels
      }
      labelSpreadingRec(laplacian, newLabels.cache(), iteration + 1)
    }

    val degrees = distances.toIndexedRowMatrix.rows.map { case row =>
      (row, row.vector.toArray.sum)
    }.cache
    val laplacian = new CoordinateMatrix(degrees.cartesian(degrees).map {case ((IndexedRow(i, v1), d1), (IndexedRow(j, v2), d2)) =>
      new MatrixEntry(i, j, (i, j) match {
        case (`j`, _) if v1.numNonzeros > 1 => $(alpha)
        case (`i`, `j`) if v1(j.toInt) != 0 && v2(i.toInt) != 0 => $(alpha) / math.sqrt(d1 * d2)
        case _ => 0.0d
      })
    }.filter {_.value !== 0.0d +- 0.00000000001d}, distances.numRows, distances.numCols).toBlockMatrix(toLabel.rowsPerBlock, toLabel.colsPerBlock).cache()
    labelSpreadingRec(laplacian, toLabel, 0)
  }

  override def copy(extra: ParamMap): LabelSpreadingClassifier = {
    defaultCopy(extra)
  }

}