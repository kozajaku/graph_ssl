package cz.cvut.fit.palicand.vocloud.ssl.utils
import org.apache.spark.mllib.linalg._
import breeze.linalg.{Vector => VB, DenseVector => DVB, SparseVector => SBV}
/**
  * Created by palickaa on 23/03/16.
  */
object VectorUtils {
  def toBreeze(v: Vector) : VB[Double] = {
    v match  {
      case DenseVector(values: Array[Double]) => new DVB[Double](values)
      case SparseVector(size: Int, indices: Array[Int], values: Array[Double]) => new SBV[Double](indices, values, size)
      case _ =>
        throw new UnsupportedOperationException(
          s"Do not support conversion from type ${v}.")
    }
  }
}
