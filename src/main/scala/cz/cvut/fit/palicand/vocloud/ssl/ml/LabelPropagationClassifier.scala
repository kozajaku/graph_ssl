package cz.cvut.fit.palicand.vocloud.ssl.ml

import breeze.linalg.sum
import breeze.numerics.abs
import cz.cvut.fit.palicand.vocloud.ssl.utils.MatrixUtils
import org.apache.spark.Logging
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

import scala.annotation.tailrec

/**
  * Created by palickaa on 08/03/16.
  */

private[ml] trait KnnKernel extends Logging {
  def knnKernel(dataset: RDD[Vector], distance: (Vector, Vector) => Double, k: Int): BlockMatrix = {
    case class Neighbour(vec: Long, distance: Double)
    val length = dataset.count()
    val dsWithIndex = dataset.zipWithIndex()
    val pairs = dsWithIndex.cartesian(dsWithIndex)
    val distances = pairs.map {
      case ((p1, index1), (p2, index2)) => (index1, Neighbour(index2, distance(p1, p2)))
    }
    val neighbours = distances.aggregateByKey(List[Neighbour]())({ case (xs: List[Neighbour], x) =>
      (x :: xs).sortBy {
        _.distance
      }.take(k)
    }, { case (xs: List[Neighbour], ys: List[Neighbour]) =>
      (xs ::: ys).sortBy({
        _.distance
      }).take(k)
    }).map { case (idx, xs) =>
      logDebug(s"${idx} - ${xs}")
      IndexedRow(idx, Vectors.sparse(length.toInt, xs.map { case Neighbour(nIdx, dist) => (nIdx.toInt, dist) }))
    }

    new IndexedRowMatrix(neighbours, dataset.count(), length.toInt).toBlockMatrix()
  }
}

final class LabelPropagationClassifier(override val uid: String)
  extends GraphClassificator[Vector, LabelPropagationClassifier, LabelPropagationModel](uid) with KnnKernel {
  def this() = this(Identifiable.randomUID("lpc"))

  override def copy(extra: ParamMap): LabelPropagationClassifier = defaultCopy(extra)

  override protected def train(dataset: DataFrame): LabelPropagationModel = {
    logDebug("Starting training")
    val labeledPoints = extractLabeledPoints(dataset)
    val distances = $(neighbourhoodKernel) match {
      case "knn" => {
         knnKernel(labeledPoints.map {_.features}, Vectors.sqdist, $(kNeighbours))
      }
    }
    val encoder = new OneHotEncoder()
    encoder.setInputCol($(labelCol))
    encoder.setOutputCol("labelIndex")
    // setDropLast encodes the last label as a zero vector. Thus, if we store an unknown label as label number
    // (n_labels), then we get what we need.
    encoder.setDropLast(true)
    val labelColStr = $(labelCol)
    val labeledRows = dataset.select(labelColStr).where(s"$labelColStr != ${dataset.select(s"$labelColStr").distinct().count()}").count()
    val toLabel = new IndexedRowMatrix(encoder.transform(dataset).select("labelIndex").rdd.zipWithIndex.map { case (Row(v: Vector), i) => new IndexedRow(i, v) }).toBlockMatrix()
    logDebug(s"Label matrix:\n${toLabel.toLocalMatrix()}")
    val labels = labelPropagation(distances, toLabel, labeledRows)
    new LabelPropagationModel(labels)
  }

  private def labelPropagation(weights: BlockMatrix, labels: BlockMatrix, labeledPoints: Long): BlockMatrix = {
    @tailrec
    def labelPropagationRec(transitionMatrix: BlockMatrix, labels: BlockMatrix, iteration: Int): BlockMatrix = {
      if (iteration > $(maxIterations)) {
        return labels
      }
      logDebug(s"${labels.toLocalMatrix()}")
      val oldLabelsRDD = MatrixUtils.blockToCoordinateMatrix(labels).toIndexedRowMatrix.rows
      val newLabelsRDD = MatrixUtils.blockToCoordinateMatrix(transitionMatrix.multiply(labels)).toIndexedRowMatrix().rows.repartition(oldLabelsRDD.getNumPartitions)
      logDebug(s"${oldLabelsRDD.count()} ${oldLabelsRDD.getNumPartitions} ${newLabelsRDD.count()} ${newLabelsRDD.getNumPartitions}")
      val newLabels = new IndexedRowMatrix(newLabelsRDD.zip(oldLabelsRDD).map {
        case (IndexedRow(iNew, vNew), oldRow) =>
          if (iNew < labeledPoints) {
            oldRow
          }
          else {
            new IndexedRow(iNew, vNew)
          }
      }).toBlockMatrix()
      if (has_converged(labels, newLabels, 0.0003)) {
        return newLabels
      }
      labelPropagationRec(transitionMatrix, transitionMatrix.multiply(labels), iteration + 1)
    }
    val transitionMatrix = new IndexedRowMatrix(weights.toIndexedRowMatrix.rows.map { (row) =>
      IndexedRow(row.index, Vectors.sparse(row.vector.size, Array(row.index.toInt), Array(1 / row.vector.toArray.sum)))
    }).toBlockMatrix.multiply(weights)
    logDebug(s"${weights.toLocalMatrix()}\n\n${transitionMatrix.toLocalMatrix()}")
    labelPropagationRec(transitionMatrix, labels, 0)
  }

  private def has_converged(oldLabels: BlockMatrix, newLabels: BlockMatrix, tolerance: Double): Boolean = {
    val oldLabelsRDD = oldLabels.blocks
    val newLabelsRDD = newLabels.blocks
    oldLabels.blocks.zip(newLabels.blocks).map({ case (((i1, j1), m1), ((i2, j2), m2)) =>
      val substracted = abs((MatrixUtils.toBreeze(m1) - MatrixUtils.toBreeze(m2)).toDenseMatrix)
      sum(substracted)
    }).reduce(_ + _) <= tolerance
  }


}


private[ml] object LabelPropagationClassifier {

}