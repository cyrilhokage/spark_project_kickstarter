package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")

/*******************************************************************************
  *
  *       TP 2
  *
  *       - Charger un fichier csv dans un dataFrame
  *
  ********************************************************************************/

    val df:DataFrame=spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("/cal/homes/cnouboue/Desktop/TP_SPARK_PROJET/TP2_PROJET/spark_project_kickstarter/data/train.csv")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

  /** Affichez un extrait du DataFrame sous forme de tableau :
    df.show()
    */

  // df.printSchema()

  // Assignez le type Int aux colonnes qui vous semblent contenir des entiers :

  val dfCasted: DataFrame = df
    .withColumn("goal", $"goal".cast("Int"))
    .withColumn("deadline" , $"deadline".cast("Int"))
    .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
    .withColumn("created_at", $"created_at".cast("Int"))
    .withColumn("launched_at", $"launched_at".cast("Int"))
    .withColumn("backers_count", $"backers_count".cast("Int"))
    .withColumn("final_status", $"final_status".cast("Int"))

    // dfCasted.printSchema()

    // Affichez une description statistique des colonnes de type Int :
    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

  }
}
