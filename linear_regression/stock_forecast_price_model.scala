import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint

object Linear_regression{

def main(args:Array[String]){

    val sc = new SparkContext("local", "LR", System.getenv("SPARK_HOME"), SparkContext.jarOfClass(this.getClass))

    // Load and parse the data
    val data = sc.textFile("spark_train")
    val parsedData = data.map { line =>
          val parts = line.split(',')
          LabeledPoint(parts(0).toDouble, parts(1).split(',').map(x => x.toDouble).toArray)
    }

    val data1 = sc.textFile("spark_test")
    val parsedData1 = data1.map { line =>
            val parts1 = line.split(',')
            LabeledPoint(parts1(0).toDouble, parts1(1).split(',').map(x => x.toDouble).toArray)
    }

    // Building the model
    val numIterations = 30
    val model = LinearRegressionWithSGD.train(parsedData, numIterations)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = parsedData1.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val average_error = valuesAndPreds.map{ case(v, p) => math.abs(v - p)}.reduce(_ + _)/valuesAndPreds.count
    println("Training Error = " + average_error)
 }

}
