package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._

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

    import spark.implicits._  // <-- Add This

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

    // On supprime la colonne 'disable_communication' parce que la grande majorité de ces valeurs sont 'falses'
    val df2: DataFrame = dfCasted.drop("disable_communication")

    // pour enlever les données du futur
    // on retire les colonnes backers_count et state_changed_at

    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    //Fonction pour nettoyer la colonne 'country'
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    //Fonction pour nettoyer la colonne 'currency'
    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    //On cree des UDF
    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    //Dataframe avec les colones 'country', et 'currency' nettoyées
    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    //Convertion des colonnes dates en timeStamp
    val dfDate_tmsp: DataFrame = dfCountry
      .withColumn("launched_at", from_unixtime($"launched_at"))
      .withColumn("created_at", from_unixtime($"created_at"))
      .withColumn("deadline", from_unixtime($"deadline"))

    //Calcul des nouvelles colonnes 'hours_prepa', 'days_campaign'
    val dfDate: DataFrame = dfDate_tmsp
      .withColumn("days_campaign", datediff($"deadline", $"launched_at"))
      .withColumn("hours_prepa", datediff($"launched_at", $"created_at"))
      .drop("deadline", "launched_at", "created_at")

    // On transfrome tous les caracteres des colonnes textes en minuscules
    // et on les concatene en une seule colonne
    val dfText: DataFrame = dfDate
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))
      .withColumn("name_desc", concat($"name", lit(" "), $"desc"))
      .withColumn("text", concat($"name_desc", lit(" "), $"keywords"))
      .drop("name", "desc", "keywords", "name_desc")


    //Fonction pour nettoyer les valeurs nulles des colonnes
    // 'days_campaign', 'hours_prepa' et 'goal'
    def clean_null_int(int: Int): Int = {
      if (int ==null ) // or int isNaN ?
        -1
      else
        int
    }

    //Fonction pour nettoyer les valeurs nulles des colonnes
    // 'country2', 'currency2'
    def clean_null_string(string: String): String = {
      if (string == null)
        "unknown"
      else
        string
    }

    //On cree des UDF
    val clean_null_intUdf = udf(clean_null_int _)
    val clean_null_stringUdf = udf(clean_null_string _)

    //On nettoie les valeurs null dans le dataframe
    val dfClean: DataFrame = dfText
      .withColumn("days_campaign2", clean_null_intUdf($"days_campaign"))
      .withColumn("hours_prepa2", clean_null_intUdf($"hours_prepa"))
      .withColumn("goal2", clean_null_intUdf($"goal"))
      .withColumn("country3", clean_null_stringUdf($"country2"))
      .withColumn("currency3", clean_null_stringUdf($"currency2"))
      .drop("days_campaign", "hours_prepa", "goal","country2", "currency2")

    //On sauvegarde le dataframe propre
    dfClean.write.parquet("./data/dfClean")

    //Probleme au niveau des monaies ?
    // dfClean.groupBy("currency3").count.orderBy($"count".desc).show(100)

    //Probleme au niveau des pays ?
    //dfClean.groupBy("country3").count.orderBy($"count".desc).show(100)

    //dfClean.groupBy("hours_prepa2").count.orderBy($"count".desc).show(100)

  }
}
