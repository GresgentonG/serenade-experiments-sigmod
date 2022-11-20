
import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter}
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

object Preprocessing {
    case class Row(timestamp: Long, userId: Int, itemId: Int, sessionId: Int)
    def parseRating(str: String): Row = {
        val fields = str.split("\t")
        assert(fields.size == 4)
        Row(fields(0).toLong, fields(1).toInt, fields(2).toInt, fields(3).toInt)
    }

    def getList(path: String): List[Row] = {
        val src = Source
            .fromFile(path)
        val l = src.getLines()
            .map(parseRating)
            .toList
            .sortBy(r => r.timestamp)
        src.close()
        l
    }

    case class Session(begin: Long, end: Long, itemIdList: List[Int], sessionId: Int)

    def load(args: Array[String]): (List[Session], Array[List[Session]]) = {
        val full = args(0)
        val test = args(1)
        val fullList = getList(full)
        val testList = getList(test)

        val combined = (fullList ++ testList).sortBy(r => r.timestamp)
        val max = combined.last.timestamp
        val min = combined.head.timestamp

        println(s"all data, maxTs:$max, minTs:$min")

        val endOfFull = fullList.map(r => r.timestamp).max

        val sessions = combined.groupBy(r => r.sessionId).map(t => {
            val rows = t._2.sortBy(_.timestamp)
            val tss = rows.map(_.timestamp)
            val sessionId = t._1
            Session(tss.min, tss.max, rows.map(_.itemId), sessionId)
        }).toList.sortBy(s => s.begin)

        // split endOfFull to max into n slices
        val n = 10
        val step = (max - endOfFull) / n
        val intervals = (0 until n).map(i => (i * step + endOfFull, i * step + endOfFull + step - 1))

        val baseSessions = sessions.filter(s => s.end < endOfFull)
        val intervalSessions = Array.fill(n)(ListBuffer.empty[Session])

        for (s <- sessions) {
            for (i <- (0 until n)) {
                if (s.begin > intervals(i)._1 && s.begin < intervals(i)._2) {
                    intervalSessions(i).append(s)
                }
            }
        }

        (baseSessions, intervalSessions.map(_.toList))
    }

    def prepare(s: (List[Session], Array[List[Session]]), dst: String): Unit = {
        val n = 10

        for (i <- (0 until n)) {
            val train = ListBuffer.empty[Row]
            val test = mutable.HashMap.empty[Int, List[Int]]
            val test2 = ListBuffer.empty[List[Int]]

            train.appendAll(s._1.flatMap(ss => ss.itemIdList.map(id => Row(ss.begin, ss.sessionId, id, ss.sessionId))))

            val currentIntervalSessions = s._2(i)
            for (ss <- currentIntervalSessions) {
                val length = ss.itemIdList.length
                if (length >= 2) {
                    val (firstHalf, secondHalf) = ss.itemIdList.splitAt(length/ 2)
                    train.appendAll(firstHalf.map(id => Row(ss.begin, ss.sessionId, id, ss.sessionId)))
                    test(ss.sessionId) = secondHalf
                    test2.append(List(ss.sessionId) ++ secondHalf)
                }
            }

            val trainWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(s"$dst/train${i}.txt")))
            val testWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(s"$dst/test${i}.txt")))
            val test2Writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(s"$dst/testSecondHalf${i}.txt")))



            train.foreach(l => trainWriter.write(s"${l.timestamp},${l.sessionId},${l.itemId}\n"))
            test.foreach(t => testWriter.write(s"${t._1},${t._2.mkString(",")}\n"))
            test2.foreach(t => test2Writer.write(s"${t.mkString(",")}\n"))

            trainWriter.flush()
            trainWriter.close()
            testWriter.flush()
            testWriter.close()
            test2Writer.flush()
            test2Writer.close()
        }
    }

    def main(args: Array[String]): Unit = {
        val loaded = load(args)
        prepare(loaded, "/path/to/dst")
    }
}
