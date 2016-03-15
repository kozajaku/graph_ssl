package cz.cvut.fit.palicand.vocloud.ssl.utils

import org.apache.spark.sql.DataFrame

/**
  * Created by palickaa on 13/03/16.
  */
object DataframeUtils {

  def numberOfClasses(df: DataFrame, classCol: String): Long = {
    df.select(s"$classCol").distinct().count() - 1
  }

  def numberOfLabeledRows(df: DataFrame, classCol: String) : Long = {
    df.select(classCol).where(s"$classCol != ${numberOfClasses(df, classCol)}").count()
  }
}
