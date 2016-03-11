package cz.cvut.fit.palicand.vocloud.ssl.utils

import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, MatrixEntry, CoordinateMatrix}
import org.apache.spark.mllib.linalg.{SparseMatrix, DenseMatrix, Matrix}
import breeze.linalg.{Matrix => BM, DenseMatrix => BDM, CSCMatrix => BSM}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by palickaa on 10/03/16.
  */
object MatrixUtils {
  def toBreeze(m: Matrix): BM[Double] = {
    m match {
      case dm: DenseMatrix =>
        if (dm.isTransposed) {
          new BDM[Double](dm.numRows, dm.numCols, dm.values)
        } else {
          val breezeMatrix = new BDM[Double](dm.numCols, dm.numRows, dm.values)
          breezeMatrix.t
        }

      case sm: SparseMatrix =>
        if (!sm.isTransposed) {
          new BSM[Double](sm.values, sm.numRows, sm.numCols, sm.colPtrs, sm.rowIndices)
        } else {
          val breezeMatrix = new BSM[Double](sm.values, sm.numCols, sm.numRows, sm.colPtrs, sm.rowIndices)
          breezeMatrix.t
        }
      case _ =>
        throw new UnsupportedOperationException(
          s"Do not support conversion from type ${m}.")
    }
  }

  def fromBreeze(breeze: BM[Double]): Matrix = {
    breeze match {
      case dm: BDM[Double] =>
        new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)
      case sm: BSM[Double] =>
        // There is no isTranspose flag for sparse matrices in Breeze
        new SparseMatrix(sm.rows, sm.cols, sm.colPtrs, sm.rowIndices, sm.data)
      case _ =>
        throw new UnsupportedOperationException(
          s"Do not support conversion from type ${breeze.getClass.getName}.")
    }
  }

  def blockToCoordinateMatrix(m: BlockMatrix): CoordinateMatrix = {
    val entryRDD = m.blocks.flatMap { case ((blockRowIndex, blockColIndex), mat) =>
      val rowStart = blockRowIndex.toLong * m.rowsPerBlock
      val colStart = blockColIndex.toLong * m.colsPerBlock
      val entryValues = new ArrayBuffer[MatrixEntry]()
      val bmMat = toBreeze(mat)
      bmMat.foreachPair { case ((i, j), v) =>
        if (j == 0.0)   entryValues.append(new MatrixEntry(rowStart + i, colStart + j, v))
        else if (v != 0.0) entryValues.append(new MatrixEntry(rowStart + i, colStart + j, v))
      }
      entryValues
    }
    new CoordinateMatrix(entryRDD, m.numRows(), m.numCols())
  }

}
