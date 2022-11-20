/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions.col

import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}
import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.io.Source
// $example off$
import org.apache.spark.sql.SparkSession

object CFR {
    case class Rating(timestamp: Long, sessionId: Int, itemId: Int, rating: Float)
    def parseRating(str: String): Rating = {
        val fields = str.split(",")
        Rating(fields(0).toLong, fields(1).toInt, fields(2).toInt, 1)
    }

    val index = 9

    def main(args: Array[String]): Unit = {
        val spark = SparkSession
            .builder
            .appName("ALSExample")
            .master("local[*]")
            .getOrCreate()
        import spark.implicits._

        val ratings = spark.read.textFile(s"train$index.txt")
            .map(parseRating)
            .toDF()
        val als = new ALS()
            .setMaxIter(5)
            .setRegParam(0.01)
            .setUserCol("sessionId")
            .setItemCol("itemId")
            .setRatingCol("rating")
        val model = als.fit(ratings)
        model.setColdStartStrategy("drop")
        val users = spark.read.textFile(s"test$index.txt").map(l => l.split(",").head).toDF("sessionId")
        val userSubsetRecs = model.recommendForUserSubset(users, 20)
        userSubsetRecs.map(r => s"${r.get(0)},${r.get(1)}")
        userSubsetRecs.show()

        val source = Source.fromFile(s"testSecondHalf$index.txt")
        val sessionIdToFirstHalf = source.getLines().map(l => l.split(",")).map(t => (t.head.toInt, t.tail)).toMap

        userSubsetRecs.printSchema
        val sessionIds: List[Int] = userSubsetRecs.select("sessionId").collectAsList().map(a => a.get(0).asInstanceOf[Int]).toList
        val recs = userSubsetRecs.withColumn("recommendationIds", col("recommendations.itemId")).select("recommendationIds").collectAsList().map(a => a.getSeq(0).mkString(","))


        val predictionWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(s"/tmp/prediction$index.txt")))
        sessionIds.zip(recs).foreach(t => predictionWriter.write(s"${t._2};${sessionIdToFirstHalf(t._1).mkString(",")}\n"))

        predictionWriter.flush()
        predictionWriter.close()

        spark.stop()
    }
}
