import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier, NaiveBayes, DecisionTreeClassifier, OneVsRest}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.feature.{StringIndexer, IndexToString}
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.types._
// spark-shell --driver-memory 6G -i windowingByExploding.scala 

spark.sparkContext.setCheckpointDir("checked_dfs")

val in_file = "sample.txt"
val stratified = true
val wsize = 7
val ngram = 3
val minF = 2

val windUdf = udf{s: String => s.sliding(ngram).toList.sliding(wsize).toList}
val get_mid = udf{s: Seq[String] => s(s.size/2)}
val rm_punct = udf{s: String => s.replaceAll("""([\p{Punct}|¿|\?|¡|!]|\p{C}|\b\p{IsLetter}{1,2}\b)\s*""", "")}

var df = spark.read.text(in_file).withColumn("value", rm_punct('value))

df = df.withColumn("char_nGrams", windUdf('value)).withColumn("word_winds", explode($"char_nGrams")).withColumn("word", get_mid('word_winds))
val indexer = new StringIndexer().setInputCol("word").setOutputCol("label")
df = indexer.fit(df).transform(df)

val hashingTF = new HashingTF().setInputCol("word_winds").setOutputCol("freqFeatures")
df = hashingTF.transform(df)
val idf = new IDF().setInputCol("freqFeatures").setOutputCol("features")
df = idf.fit(df).transform(df)
// Remove word whose freq is less than minF
var counts = df.groupBy("label").count.filter(col("count") > minF).orderBy(desc("count")).withColumn("id", monotonically_increasing_id())
var filtro = df.groupBy("label").count.filter(col("count") <= minF)
df = df.join(filtro, Seq("label"), "leftanti")

var dfs = if(stratified){
// Create stratified sample 'dfs' 
        var revs = counts.orderBy(asc("count")).select("count").withColumn("id", monotonically_increasing_id())
        revs = revs.withColumnRenamed("count", "ascc")
// Weigh the labels (linearly) inversely ("ascc") proportional NORMALIZED weights to word ferquency
// Interestingly, this gives a sample 20% of the size of the whole dataset
        counts = counts.join(revs, Seq("id"), "inner").withColumn("weight", col("ascc")/df.count)
        val minn = counts.select("weight").agg(min("weight")).first.getDouble(0) - 0.01
        val maxx = counts.select("weight").agg(max("weight")).first.getDouble(0) - 0.01
        counts = counts.withColumn("weight_n", (col("weight") - minn) / (maxx - minn))
        counts = counts.withColumn("weight_n", when(col("weight_n") > 1.0, 1.0).otherwise(col("weight_n")))
        var fractions = counts.select("label", "weight_n").rdd.map(x => (x(0), x(1).asInstanceOf[scala.Double])).collectAsMap.toMap
        df.stat.sampleBy("label", fractions, 36L).select("features", "word_winds", "word", "label")
        }else{ df }
dfs = dfs.checkpoint()
//val layers = Array[Int](hashingTF.getNumFeatures, df.select("label").distinct.count.toInt)
//val lr = new LogisticRegression().setRegParam(0.01)
//val classifier = new LogisticRegression().setRegParam(0.01).setMaxIter(10).setTol(1E-6).setFitIntercept(false)
//val classifier = new NaiveBayes()
val classifier = new DecisionTreeClassifier()
var lr = new OneVsRest().setClassifier(classifier)
//val lr = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(20).setSeed(1234L).setMaxIter(100)
//val lr = new RandomForestClassifier().setNumTrees(2)

val Array(tr, ts) = dfs.randomSplit(Array(0.7, 0.3), seed = 12345)
val training = tr.select("word_winds", "features", "label", "word") 
val test = ts.select("word_winds", "features", "label", "word")
val model = lr.fit(training)

def mapCode(m: scala.collection.Map[Any, String]) = udf( (s: Double) =>
                m.getOrElse(s, "")
        )
var labels = training.select("label", "word").distinct.rdd.map(x => (x(0), x(1).asInstanceOf[String])).collectAsMap
var predictions = model.transform(test)
predictions = predictions.withColumn("pred_word", mapCode(labels)('prediction))
predictions.write.format("csv").save("spark_predictions")
