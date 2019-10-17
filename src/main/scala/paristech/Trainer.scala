package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, OneHotEncoderEstimator, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")

    /* On importe le dataframe du format parquet */
    val df_prepared = spark.read.parquet("./data/prepared_trainingset")

    df_prepared.printSchema

    /* Stage 1 : Tokenisation */
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens_raw")

        /* On applique les modifs */
   // val df_tokenized: DataFrame = tokenizer.transform(df_prepared)

    /* Stage 2: Stop-word remover */
    val sw_remover  = new StopWordsRemover()
      .setInputCol("tokens_raw")
      .setOutputCol("tokens")

        /* On applique les modifs */
   // val df_filtered: DataFrame = sw_remover.transform(df_tokenized)

    /* Stage 3 computer la partie TF */

    // fit a CountVectorizerModel from the corpus
    val count_vect = new CountVectorizer()
      .setInputCol("tokens")
      .setOutputCol("features")
      // .fit(df_filtered)


  /*
    val cvm = new CountVectorizerModel(Array("a", "b", "c"))
      .setInputCol("tokens")
      .setOutputCol("features") */

    /*
    // fit a CountVectorizerModel from the corpus
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("tokens")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df_filtered) */

    // val df_countVect : DataFrame = count_vect.fit(df_filtered).transform(df_filtered)
    //df_countVect.show(5)


    // Stage 4 : Compute IDF
    val idf = new IDF().setInputCol("features").setOutputCol("tf_idf")
    // val idfModel = idf.fit(df_countVect)

    // val rescaledData = idfModel.transform(df_countVect)

    /*
    //On affiche ce qu'on a
    rescaledData.printSchema
    rescaledData.select("text", "tf_idf").show(5, false)

     */

    println("\n END TEXT PREPROPCESSING \n ")

    /*  Conversion des variables catégorielles en variables numériques */

    //Stage 5 :  convertir country2 en quantités numériques
    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      //.fit(rescaledData)

   // val country_ind_data = country_indexer.transform(rescaledData)

    /* On affiche pour vérifier
    country_ind_data.groupBy("country2", "country_indexed")
      .count()
      .show()
*/
    // Stage 6 :

    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      //.fit(country_ind_data)

    //val curr_ind_data = currency_indexer.transform(country_ind_data)

    /* On affiche pour vérifier
    curr_ind_data.groupBy("currency2", "currency_indexed")
      .count()
      .show() */

  // Stage 7 & 8 : One-hot encoder
    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array(country_indexer.getOutputCol, currency_indexer.getOutputCol))
      .setOutputCols(Array("country_onehot", "currency_onehot"))
   //   .fit(curr_ind_data)

    // val encoded_data = encoder.transform(curr_ind_data)

    // Stage 9 : assembler toutes les features en un unique vecteur
    // On recupere le nom des colonnes dans une liste
    val featureCols = Array("tf_idf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot")

    // On les assemble
    val featureAssembler = new VectorAssembler().setInputCols(featureCols)
                                  .setOutputCol("assembled_features")  // Output column features already exists.

    // val dataWithFeatures = featureAssembler.transform(encoded_data)

    //Stage 10 : créer/instancier le modèle de classification : regression logistique

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("assembled_features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    // Learn a LogisticRegression model. This uses the parameters stored in lr.
   // val model1 = lr.fit(dataWithFeatures)
   // println(s"Model 1 was fit using parameters: ${model1.parent.extractParamMap}")

    //Stage 11 : Creation du pipeline
    val stages = Array(tokenizer, sw_remover, count_vect, idf, country_indexer, currency_indexer,
      encoder, featureAssembler, lr)

    val pipeline = new Pipeline()
      .setStages(stages)

    // Fit the pipeline to training documents.
    val model = pipeline.fit(df_prepared)


    // Now we can optionally save the fitted pipeline to disk
    model.write.overwrite().save("./model/spark-logistic-regression-model")

    /*
    // We can also save this unfit pipeline to disk
     pipeline.write.overwrite().save("./model/unfit-lr-model")

     // And load it back in during production
     val sameModel = PipelineModel.load("./model/spark-logistic-regression-model")
     */

  val transform_data = model.transform(df_prepared)


    transform_data.select("name", "final_status", "predictions")
      .show(5)

    //On affiche les résultats du modele
    transform_data.groupBy("final_status", "predictions").count.show()


  }
}
