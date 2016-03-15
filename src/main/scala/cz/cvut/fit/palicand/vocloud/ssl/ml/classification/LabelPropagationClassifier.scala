package cz.cvut.fit.palicand.vocloud.ssl.ml.classification

import breeze.linalg.sum
import breeze.numerics.abs
import cz.cvut.fit.palicand.vocloud.ssl.utils.{DataframeUtils, MatrixUtils, VectorUtils}
import org.apache.spark.Logging
import org.apache.spark.ml.classification.{ClassificationModel, Classifier, ProbabilisticClassificationModel}
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import scala.annotation.tailrec

/**
  * Created by palickaa on 08/03/16.
  */


trait GraphParams extends Params {
  final val neighbourhoodKernel = new Param[String](this, "neighKernel", "Kernel to use for getting neighbourhors",
    ParamValidators.inArray(Array("knn")))
  setDefault(neighbourhoodKernel -> "knn")

  final val kNeighbours = new IntParam(this, "kNeighbours", "K in KNN", ParamValidators.gtEq(1))
  setDefault(kNeighbours -> 1)

  final val maxIterations = new IntParam(this, "maxIterations", "How long should we iterate")
  setDefault(maxIterations -> 100)
}

abstract class GraphClassifier(override val uid: String)
  extends Classifier[Vector, GraphClassifier, LabelPropagationModel] with GraphParams with KnnKernel {

  def setKNeighbours(k: Int): GraphClassifier = {
    set(kNeighbours, k).asInstanceOf[GraphClassifier]
  }

  def setMaxIterations(iterations: Int): GraphClassifier = {
    set(maxIterations, iterations).asInstanceOf[GraphClassifier]
  }

  def setNeighbourhoodKernel(kernel: String): GraphClassifier = {
    set(neighbourhoodKernel, kernel).asInstanceOf[GraphClassifier]
  }

  protected def computeLabelProbabilities(distances: BlockMatrix, labels: BlockMatrix, numberOfLabeled: Long) : BlockMatrix

  override def train(dataset: DataFrame) : LabelPropagationModel = {
    val labeledPoints = extractLabeledPoints(dataset)
    val distances = $(neighbourhoodKernel) match {
      case "knn" => {
        knnKernel(dataset, $(featuresCol), Vectors.sqdist, $(kNeighbours)).cache()
      }
    }
    val encoder = new OneHotEncoder()
    encoder.setInputCol($(labelCol))
    encoder.setOutputCol("labelIndex")
    // setDropLast encodes the last label as a zero vector. Thus, if we store an unknown label as label number
    // (n_labels), then we get what we need.
    encoder.setDropLast(true)
    val labeledRows = DataframeUtils.numberOfLabeledRows(dataset, $(labelCol))
    logDebug(s"Number of labeled rows: $labeledRows")
    val toLabel = new IndexedRowMatrix(encoder.transform(dataset).select("rowNo", "labelIndex").rdd.map {
      case (Row(i: Long, v: Vector)) =>
        new IndexedRow(i, v.toDense)
    }).toBlockMatrix().cache()
    val labels = computeLabelProbabilities(distances, toLabel, labeledRows)
    new LabelPropagationModel(labeledPoints, distances, labels,
      new DenseVector(MatrixUtils.blockToCoordinateMatrix(labels).toIndexedRowMatrix.rows.sortBy(_.index).map { row => row.vector.argmax.toDouble }.collect()),
      $(kNeighbours))
  }

  protected def hasConverged(oldLabels: BlockMatrix, newLabels: BlockMatrix, tolerance: Double): Boolean = {
    /*val delta: Double = oldLabels.blocks.join(newLabels.blocks).map({ case ((i1, j1), (m1, m2)) =>
      val substracted = abs(MatrixUtils.toBreeze(m1).toDenseMatrix - MatrixUtils.toBreeze(m2).toDenseMatrix)
      sum(substracted)
    }).mean()*/
    val oldRows = oldLabels.toIndexedRowMatrix.rows.map { row => (row.index, row) }
    val newRows = newLabels.toIndexedRowMatrix.rows.map { row => (row.index, row) }

    val delta = oldRows.fullOuterJoin(newRows).map { case (rowNo, (r1, r2)) =>
      val v1 = r1 match {
        case Some(row) => row.vector
        case None => Vectors.zeros(oldLabels.numCols().toInt)
      }
      val v2 = r2 match {
        case Some(row) => row.vector
        case None => Vectors.zeros(oldLabels.numCols().toInt)
      }
      val substracted = abs(VectorUtils.toBreeze(v1).toDenseVector - VectorUtils.toBreeze(v2).toDenseVector)
      sum(substracted)
    }.mean
    assert(!delta.isNaN && !delta.isInfinity)
    logDebug(s"Delta is $delta")
    delta <= tolerance
  }
}


private[ml] trait KnnKernel extends Logging {
  def knnKernel(dataset: DataFrame, featCol: String, distance: (Vector, Vector) => Double, k: Int): BlockMatrix = {
    val dsSize = dataset.count()
    val knn = new KNN().setFeaturesCol(featCol).setAuxCols(("rowNo" +: featCol +: Nil).toArray).setK(k).setTopTreeSize((0.5 * dsSize).ceil.toInt).setBufferSizeSampleSizes(Array(50, 100))
    val knnModel = knn.fit(dataset)
    new CoordinateMatrix(knnModel.transform(dataset).select("rowNo", featCol, "neighbors").rdd.flatMap {
      case (Row(rowIndex: Long, vec: Vector, neighbours: Seq[_])) =>
        neighbours.flatMap { case (neighbourRow: Row) =>
          //TODO FIND SMARTER WAY TO DO IT
          val dist = distance(vec, neighbourRow.getAs[Vector](1)) + 0.0000001 //to fix an issue when we have distance == 0

          MatrixEntry(rowIndex, neighbourRow.getLong(0), dist) :: MatrixEntry(neighbourRow.getLong(0), rowIndex, dist) :: Nil
        }
    }).toBlockMatrix
  }
}

final class LabelPropagationClassifier(override val uid: String)
  extends GraphClassifier(uid)  {
  def this() = this(Identifiable.randomUID("lpc"))

  override def copy(extra: ParamMap): LabelPropagationClassifier = defaultCopy(extra)


  override def computeLabelProbabilities(weights: BlockMatrix, labels: BlockMatrix, labeledPoints: Long): BlockMatrix = {
    @tailrec
    def labelPropagationRec(transitionMatrix: BlockMatrix, labels: BlockMatrix, iteration: Int): BlockMatrix = {
      if (iteration > $(maxIterations)) {
        return labels
      }
      val oldLabelsRDD = labels.toIndexedRowMatrix.rows.sortBy(_.index).coalesce(32)
      val newLabelsRDD = MatrixUtils.blockToCoordinateMatrix(transitionMatrix.multiply(labels)).toIndexedRowMatrix.rows.sortBy(_.index).coalesce(32)
      val newLabels = new IndexedRowMatrix(oldLabelsRDD.filter { case IndexedRow(i, v) =>
        i < labeledPoints
      } ++
        newLabelsRDD.filter { case IndexedRow(i, v) =>
          i >= labeledPoints
        }).toBlockMatrix()
      //assert(MatrixUtils.hasOnlyValidElements(newLabels))
      if (hasConverged(labels, newLabels, 0.001)) {
        return newLabels
      }
      labelPropagationRec(transitionMatrix, newLabels.cache(), iteration + 1)
    }


    val rowD = new IndexedRowMatrix(MatrixUtils.blockToCoordinateMatrix(weights).toIndexedRowMatrix.rows.map { (row) =>

      //assert(!inverse.isNaN)
      IndexedRow(row.index, Vectors.sparse(row.vector.size, Array(row.index.toInt), Array(1.0 / row.vector.toArray.sum)))
    })

    //assert(MatrixUtils.hasOnlyValidElements(rowD))

    val d = rowD.toBlockMatrix
    //assert(MatrixUtils.hasOnlyValidElements(d))
    val transitionMatrix = d.multiply(weights).cache()
    //assert(MatrixUtils.hasOnlyValidElements(transitionMatrix))
    labelPropagationRec(transitionMatrix, labels, 0)
  }


}

class LabelPropagationModel(override val uid: String, val points: RDD[LabeledPoint], val weightMatrix: BlockMatrix, val labelWeights: BlockMatrix,
                            val labels: Vector, val k: Int) extends ProbabilisticClassificationModel[Vector, LabelPropagationModel]
  with Serializable with KnnKernel with GraphParams {
  override def numClasses: Int = {
    labelWeights.numCols().toInt
  }

  private def knnOnePoint(features: Vector): SparseVector = {
    val distances = points.zipWithIndex().map { case (point, index) =>
      (index, Vectors.sqdist(point.features, features))
    }.sortBy { case (_, distance) => distance }.take(k)
    new SparseVector(weightMatrix.numRows().toInt, distances.map { case (index, _) => index.toInt },
      distances.map { case (_, v) => v })
  }

  override protected def predictRaw(features: Vector): Vector = {
    val distances = knnOnePoint(features)
    val indices = distances.indices.toSet
    val distanceSum = distances.toArray.sum
    val labelProba = MatrixUtils.blockToCoordinateMatrix(labelWeights).toIndexedRowMatrix.rows.filter { row => indices.contains(row.index.toInt) }.collect()
    new DenseVector(labelProba.zip(distances.values).map { case (pointLabels, pointDistance) =>
      val weightedProbabilities = pointLabels.vector.toArray.map {
        probability => probability * pointDistance
      }
      weightedProbabilities
    }.reduce { (left, right) =>
      left.zip(right).map { case (a, b) => (a + b) / distanceSum }
    })
  }

  override def copy(extra: ParamMap): LabelPropagationModel = defaultCopy(extra)

  def this(points: RDD[LabeledPoint], weightMatrix: BlockMatrix, labelWeights: BlockMatrix, labels: Vector, k: Int) = this(Identifiable.randomUID("lpcm"), points,
    weightMatrix, labelWeights, labels, k)

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction
  }
}