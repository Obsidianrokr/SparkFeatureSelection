/**
 * Created by vladimir on 12/23/14.
 */

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.stat._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint







object simpleApp {
  def main(args: Array[String]) {
    val dataFile = "data/sample_10k_rows_head.csv"
    val conf = new SparkConf().setAppName("SparkFeatureSelection").setMaster("local[4]")
    val sc = new SparkContext(conf)

    //add code to remove first string

    val dataInstances = sc.textFile(dataFile, 4).cache()

    //add parallelize

    //Parse format

    val lPoints = dataInstances.map(str=> {
      val parsedStr = str.split(';')
      val out = parsedStr(1).toDouble
      val feat = parsedStr(2).drop(1).dropRight(1).split(',').map(_.toDouble)
      LabeledPoint(out, Vectors.dense(feat))
    })
    val dv: Vector = Vectors.dense(0.0, 0.0, 0.0)
    val model = ChiSqSelector.fit(lPoints,20)
    val filteredData = lPoints.map(lp =>
       new LabeledPoint(lp.label, model.transform(lp.features))).collect().toSet



    val numTopFeatures = 20
    val goodnessOfFitTestResult = Statistics.chiSqTest(lPoints).zipWithIndex.sortBy{ case(res, index) => -res.statistic}.take(numTopFeatures).unzip
    var i = 1
    goodnessOfFitTestResult. { result =>
      println(s"Column $i:\n$result")
      i = i + 1

    } // summary of the test
  }
}