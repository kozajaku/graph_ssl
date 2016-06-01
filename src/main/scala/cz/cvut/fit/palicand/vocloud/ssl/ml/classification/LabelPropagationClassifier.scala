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
import org.apache.spark.storage.StorageLevel
import scala.annotation.tailrec

/**
  * Created by palickaa on 08/03/16.
  */

class  MyKNN extends KNN {
  def setBufferSize(size: Double): MyKNN = {
    set(bufferSize, size).asInstanceOf[MyKNN]
  }
}

trait GraphParams extends Params {
  final val neighbourhoodKernel = new Param[String](this, "neighKernel", "Kernel to use for getting neighbourhors",
    ParamValidators.inArray(Array("knn")))
  setDefault(neighbourhoodKernel -> "knn")

  final val kNeighbours = new IntParam(this, "kNeighbours", "K in KNN", ParamValidators.gtEq(1))
  setDefault(kNeighbours -> 1)

  final val maxIterations = new IntParam(this, "maxIterations", "How long should we iterate")
  setDefault(maxIterations -> 100)
}


private[ml] trait KnnKernel extends Logging with Params {

  final val bufferSize = new DoubleParam(this, "bufferSize", "",
    ParamValidators.gt(0.0))
  setDefault(bufferSize -> 100.0)

  def setBufferSize(size: Double): KnnKernel = {
    set(bufferSize, size).asInstanceOf[KnnKernel]
  }

  final val topTreeLeafSize = new IntParam(this, "topTreeLeafSize", "",
    ParamValidators.gtEq(1))
  setDefault(topTreeLeafSize -> 5)

  def setTopTreeLeafSize(size: Int): KnnKernel = {
    set(topTreeLeafSize, size).asInstanceOf[KnnKernel]
  }

  final val subTreeLeafSize = new IntParam(this, "subTreeLeafSize", "",
    ParamValidators.gtEq(1))
  setDefault(subTreeLeafSize -> 100)

  def setSubTreeLeafSize(size: Int): KnnKernel = {
    set(subTreeLeafSize, size).asInstanceOf[KnnKernel]
  }


  final val topTreeSize = new IntParam(this, "topTreeSize", "",
    ParamValidators.gtEq(1))
  setDefault(topTreeSize -> 1000)

  def setTopTreeSize(size: Int): KnnKernel = {
    set(topTreeSize, size).asInstanceOf[KnnKernel]
  }

  def knnKernel(dataset: DataFrame, featCol: String, distance: (Vector, Vector) => Double, k: Int): BlockMatrix = {
    val dsSize = dataset.count()
    val knn = new MyKNN().setFeaturesCol(featCol).setAuxCols(("rowNo" +: featCol +: Nil).toArray).setK(k).
      setTopTreeLeafSize($(topTreeLeafSize)).
      setSubTreeLeafSize($(subTreeLeafSize)).
      setBufferSize($(bufferSize)).
      setTopTreeSize($(topTreeSize))
    val knnModel = knn.fit(dataset)
    new CoordinateMatrix(knnModel.transform(dataset).select("rowNo", featCol, "neighbors").rdd.flatMap {
      case (Row(rowIndex: Long, vec: Vector, neighbours: Seq[_])) =>
        neighbours.flatMap { case (neighbourRow: Row) =>
          //TODO FIND SMARTER WAY TO DO IT
          val dist = distance(vec, neighbourRow.getAs[Vector](1)) + 0.0000001 //to fix an issue when we have distance == 0

          MatrixEntry(rowIndex, neighbourRow.getLong(0), dist) :: MatrixEntry(neighbourRow.getLong(0), rowIndex, dist) :: Nil
        }
    }, dsSize, dsSize).toBlockMatrix(1024,
      1024)
  }
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

  /*final val sampleSize = new Param[Array[Int]](this, "sampleSize", "",
    ParamValidators.arrayLengthGt[Int](2))
  setDefault(sampleSize -> (100 to 1000 by 100).toArray)

  def setsampleSize(sizes: Array[Int]): GraphClassifier = {
    set(sampleSize, sizes).asInstanceOf[GraphClassifier]
  }*/


  protected def computeLabelProbabilities(distances: BlockMatrix, labels: BlockMatrix, numberOfLabeled: Long) : BlockMatrix

  override def train(dataset: DataFrame) : LabelPropagationModel = {
    logInfo(s"Dataset partitions: ${dataset.rdd.partitions.length}")
    logInfo(s"Dataset size: ${dataset.count()}")
    val distances = $(neighbourhoodKernel) match {
      case "knn" =>
        knnKernel(dataset, $(featuresCol), Vectors.sqdist, $(kNeighbours))
    }
    logInfo(s"Rows per block : ${distances.rowsPerBlock}, number of elements: ${distances.numRows()}")
    val encoder = new OneHotEncoder()
    encoder.setInputCol($(labelCol))
    encoder.setOutputCol("labelIndex")
    // setDropLast encodes the last label as a zero vector. Thus, if we store an unknown label as label number
    // (n_labels), then we get what we need.
    encoder.setDropLast(true)
    val numberOfClasses = DataframeUtils.numberOfClasses(dataset, "label").toInt
    val labeledRows = DataframeUtils.numberOfLabeledRows(dataset, $(labelCol))
    logDebug(s"Number of labeled rows: $labeledRows")
    val elementsPerBlock = (dataset.count() / dataset.rdd.partitions.length).toInt
    val toLabel = new IndexedRowMatrix(encoder.transform(dataset).select("rowNo", "labelIndex").rdd.map {
      case (Row(i: Long, v: Vector)) =>
        new IndexedRow(i, v.toDense)
    }, dataset.count(), numberOfClasses).toBlockMatrix(1024, 1024)
    val labels = computeLabelProbabilities(distances, toLabel, labeledRows)
    new LabelPropagationModel(extractLabeledPoints(dataset),
      distances,
      labels,
      MatrixUtils.blockToCoordinateMatrix(labels).toIndexedRowMatrix.rows.map {
        row => (row.index, (row.vector.argmax, row.vector(row.vector.argmax)))
      },
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

final class LabelPropagationClassifier(override val uid: String)
  extends GraphClassifier(uid)  {
  def this() = this(Identifiable.randomUID("lpc"))

  override def copy(extra: ParamMap): LabelPropagationClassifier = defaultCopy(extra)


  override def computeLabelProbabilities(weights: BlockMatrix, labels: BlockMatrix, labeledPoints: Long): BlockMatrix = {

    val oldLabelsRDD = labels.toIndexedRowMatrix().rows.filter { case IndexedRow(i, v) =>
      i < labeledPoints
    }.persist(StorageLevel.MEMORY_ONLY_SER)

    @tailrec
    def labelPropagationRec(transitionMatrix: BlockMatrix, labels: BlockMatrix, iteration: Int): BlockMatrix = {
      if (iteration > $(maxIterations)) {
        return labels
      }
      val newLabelsRDD = MatrixUtils.blockToCoordinateMatrix(transitionMatrix.multiply(labels)).toIndexedRowMatrix.rows
      val newLabels = new IndexedRowMatrix(oldLabelsRDD ++ newLabelsRDD.filter { case IndexedRow(i, v) =>
          i >= labeledPoints
        }, labels.numRows(), labels.numCols().toInt).toBlockMatrix(1024, 1024).persist(StorageLevel.MEMORY_ONLY_SER)
      //assert(MatrixUtils.hasOnlyValidElements(newLabels))
      if (hasConverged(labels, newLabels, 0.001)) {
        return newLabels
      }
      labelPropagationRec(transitionMatrix, newLabels, iteration + 1)
    }


    val transitionMatrix = new IndexedRowMatrix(weights.toIndexedRowMatrix.rows.map { (row) =>
      //assert(!inverse.isNaN)
      val sum = row.vector.toSparse.values.sum
      new IndexedRow(row.index, Vectors.sparse(row.vector.size, row.vector.toSparse.indices,
                     row.vector.toSparse.values.map {_ / sum}))
    }, weights.numRows(), weights.numCols().toInt).toBlockMatrix(1024, 1024).persist(StorageLevel.MEMORY_ONLY_SER)
    //assert(MatrixUtils.hasOnlyValidElements(transitionMatrix))
    labelPropagationRec(transitionMatrix, labels, 0)
  }


}

class LabelPropagationModel(override val uid: String,
                            val points: RDD[LabeledPoint],
                            val weightMatrix: BlockMatrix,
                            val labelWeights: BlockMatrix,
                            val labels: RDD[(Long, (Int, Double))],
                            val k: Int) extends ProbabilisticClassificationModel[Vector, LabelPropagationModel]
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

  def this(points: RDD[LabeledPoint], weightMatrix: BlockMatrix, labelWeights: BlockMatrix, labels: RDD[(Long, (Int, Double))],
           k: Int) = this(Identifiable.randomUID("lpcm"), points,
    weightMatrix, labelWeights, labels, k)

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction
  }
}