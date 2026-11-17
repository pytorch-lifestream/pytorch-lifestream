from pyspark.sql import SparkSession

from ptls.preprocessing.pyspark.pyspark_preprocessor import PysparkDataPreprocessor


def test_pyspark_identity_columns_without_categories():
    spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
    try:
        data = spark.createDataFrame([
            {"id": 1, "event_time": 2, "amount": 20., "labels": 101},
            {"id": 1, "event_time": 1, "amount": 10., "labels": 100},
            {"id": 2, "event_time": 1, "amount": 30., "labels": 200},
        ])
        pp = PysparkDataPreprocessor(col_id="id", col_event_time="event_time",
            event_time_transformation="none", cols_category=[],
            cols_numerical=["amount"], cols_identity=["labels"])
        pp.fit_transform(data)
        records = {row["id"]: row for row in pp.transform(data).collect()}
        assert records[1]["event_time"] == [1, 2]
        assert records[1]["labels"] == [100, 101]
        assert records[1]["amount"] == [10., 20.]
    finally:
        spark.stop()
