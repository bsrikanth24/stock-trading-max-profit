"""
PySpark Interview Questions - Complete Solutions
Working with Employee CSV Data in HDFS/ADLS
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import (
    row_number, dense_rank, col, when, broadcast, 
    coalesce, concat_ws, lit, desc, asc
)
from pyspark.sql.window import Window

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("EmployeeDataAnalysis") \
    .getOrCreate()

# ============================================================================
# QUESTION 1: Read CSV file and create DataFrame with proper schema
# ============================================================================

def question_1_read_csv_with_schema():
    """
    Read CSV file from HDFS/ADLS without header and apply manual schema.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Define schema manually
    schema = StructType([
        StructField("emp_name", StringType(), True),
        StructField("emp_id", IntegerType(), True),
        StructField("department", StringType(), True),
        StructField("salary", DoubleType(), True)
    ])
    
    # Read CSV from HDFS/ADLS
    # For HDFS: hdfs://namenode:8020/path/to/file.csv
    # For ADLS: abfss://container@storageaccount.dfs.core.windows.net/path/to/file.csv
    
    df = spark.read \
        .schema(schema) \
        .option("header", "false") \
        .option("inferSchema", "false") \
        .csv("hdfs://path/to/employees.csv")
    
    # Alternative: Using ADLS
    # df = spark.read \
    #     .schema(schema) \
    #     .csv("abfss://container@account.dfs.core.windows.net/employees.csv")
    
    print("Question 1: Read CSV with Schema")
    print("-" * 80)
    df.show()
    print(f"Schema: {df.printSchema()}")
    
    return df


# ============================================================================
# QUESTION 2: Remove rows where emp_id is NULL
# ============================================================================

def question_2_remove_null_emp_id(df):
    """
    Remove rows where emp_id is NULL.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    print("\n\nQuestion 2: Remove NULL emp_id")
    print("-" * 80)
    print("Before:")
    df.show()
    print(f"Row count: {df.count()}")
    
    # Method 1: Using filter
    df_no_null = df.filter(col("emp_id").isNotNull())
    
    # Method 2: Using dropna
    # df_no_null = df.dropna(subset=["emp_id"])
    
    print("\nAfter removing NULL emp_id:")
    df_no_null.show()
    print(f"Row count: {df_no_null.count()}")
    
    return df_no_null


# ============================================================================
# QUESTION 3: Remove duplicate records based on emp_id
# ============================================================================

def question_3_remove_duplicates(df):
    """
    Remove duplicate records based on emp_id.
    Keep the first occurrence.
    
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n)
    """
    print("\n\nQuestion 3: Remove Duplicates")
    print("-" * 80)
    print("Before:")
    df.show()
    print(f"Row count: {df.count()}")
    
    # Method 1: Using dropDuplicates (keeps first occurrence)
    df_no_dup = df.dropDuplicates(["emp_id"])
    
    # Method 2: Using window function and row_number
    # window_spec = Window.partitionBy("emp_id").orderBy("emp_name")
    # df_no_dup = df.withColumn("rn", row_number().over(window_spec)) \
    #     .filter(col("rn") == 1) \
    #     .drop("rn")
    
    print("\nAfter removing duplicates on emp_id:")
    df_no_dup.show()
    print(f"Row count: {df_no_dup.count()}")
    
    return df_no_dup


# ============================================================================
# QUESTION 4: Find the second highest salary in each department
# ============================================================================

def question_4_second_highest_salary(df):
    """
    Find the second highest salary in each department.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    print("\n\nQuestion 4: Second Highest Salary per Department")
    print("-" * 80)
    
    # Method 1: Using dense_rank window function
    window_spec = Window.partitionBy("department").orderBy(desc("salary"))
    
    df_ranked = df.withColumn(
        "rank",
        dense_rank().over(window_spec)
    )
    
    # Get second highest (rank = 2)
    df_second_highest = df_ranked.filter(col("rank") == 2) \
        .select("emp_name", "emp_id", "department", "salary", "rank")
    
    print("Second Highest Salary per Department:")
    df_second_highest.show()
    
    # Additional: Show all ranks for context
    print("\nRanked by Salary (for context):")
    df_ranked.orderBy("department", "rank").show()
    
    return df_second_highest


# ============================================================================
# QUESTION 5: Drop one column and add a new column
# ============================================================================

def question_5_drop_and_add_column(df):
    """
    Drop one column and add a new column.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    print("\n\nQuestion 5: Drop Column and Add New Column")
    print("-" * 80)
    print("Original Schema:")
    df.printSchema()
    
    # Drop 'department' column
    df_dropped = df.drop("department")
    
    print("\nAfter dropping 'department':")
    df_dropped.printSchema()
    df_dropped.show()
    
    # Add new column: salary_category
    df_enhanced = df_dropped.withColumn(
        "salary_category",
        when(col("salary") >= 100000, "Senior")
        .when(col("salary") >= 60000, "Mid-level")
        .otherwise("Junior")
    )
    
    # Alternative: Add salary_with_bonus column
    # df_enhanced = df_enhanced.withColumn(
    #     "salary_with_bonus",
    #     col("salary") * 1.1
    # )
    
    print("\nAfter adding 'salary_category' column:")
    df_enhanced.printSchema()
    df_enhanced.show()
    
    return df_enhanced


# ============================================================================
# QUESTION 6: Broadcast join with manager DataFrame
# ============================================================================

def question_6_broadcast_join(df_employees):
    """
    Perform a broadcast join with a manager DataFrame.
    
    Time Complexity: O(n) where n is size of larger table
    Space Complexity: O(n)
    """
    print("\n\nQuestion 6: Broadcast Join with Manager DataFrame")
    print("-" * 80)
    
    # Create manager DataFrame
    manager_schema = StructType([
        StructField("dept", StringType(), True),
        StructField("manager_name", StringType(), True)
    ])
    
    manager_data = [
        ("Sales", "John Smith"),
        ("Engineering", "Alice Johnson"),
        ("HR", "Bob Williams"),
        ("Finance", "Carol Davis")
    ]
    
    df_managers = spark.createDataFrame(manager_data, schema=manager_schema)
    
    print("Employees:")
    df_employees.show()
    
    print("\nManagers:")
    df_managers.show()
    
    # Broadcast join (small table is broadcasted to all worker nodes)
    df_joined = df_employees.join(
        broadcast(df_managers),
        df_employees.department == df_managers.dept,
        "left"
    ).drop("dept")
    
    print("\nAfter Broadcast Join:")
    df_joined.show()
    
    # Alternative without explicit broadcast (Spark optimizes automatically)
    # df_joined = df_employees.join(df_managers, ...)
    
    return df_joined


# ============================================================================
# QUESTION 7: Write DataFrame into 5 output files
# ============================================================================

def question_7_write_multiple_files(df):
    """
    Write DataFrame into 5 output files using repartition.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    print("\n\nQuestion 7: Write DataFrame into 5 Output Files")
    print("-" * 80)
    
    # Method 1: Repartition into 5 parts and write
    output_path = "hdfs://path/to/output/employees"
    
    df.repartition(5) \
        .write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(output_path)
    
    print(f"DataFrame written to {output_path} in 5 partition files")
    print("Files created: part-00000, part-00001, part-00002, part-00003, part-00004")
    
    # Alternative methods:
    
    # Method 2: Write with different format
    # df.repartition(5).write.mode("overwrite").parquet(output_path)
    
    # Method 3: Specify partition column
    # df.repartition(5, "department").write \
    #     .mode("overwrite") \
    #     .partitionBy("department") \
    #     .csv(output_path)
    
    # Method 4: Write each partition separately (manual control)
    # for i in range(5):
    #     df.repartition(5).write \
    #         .mode("overwrite") \
    #         .csv(f"{output_path}/part-{i}")


# ============================================================================
# BONUS: Complete End-to-End Pipeline
# ============================================================================

def end_to_end_pipeline():
    """
    Complete pipeline demonstrating all questions.
    """
    print("\n" + "="*80)
    print("PySpark Interview Questions - Complete Pipeline")
    print("="*80)
    
    # Create sample data
    schema = StructType([
        StructField("emp_name", StringType(), True),
        StructField("emp_id", IntegerType(), True),
        StructField("department", StringType(), True),
        StructField("salary", DoubleType(), True)
    ])
    
    sample_data = [
        ("Alice", 101, "Sales", 75000.0),
        ("Bob", 102, "Engineering", 95000.0),
        ("Charlie", 103, "Sales", 70000.0),
        ("David", 104, "Engineering", 100000.0),
        ("Eva", None, "HR", 65000.0),  # NULL emp_id
        ("Frank", 105, "Sales", 75000.0),  # Duplicate emp_id as Alice
        ("Grace", 101, "Engineering", 92000.0),  # Duplicate emp_id
        ("Helen", 106, "Finance", 80000.0),
        ("Ian", 107, "Sales", 68000.0),
    ]
    
    df = spark.createDataFrame(sample_data, schema=schema)
    
    print("\n1️⃣  QUESTION 1: Read CSV with Schema")
    print("-" * 80)
    df.show()
    
    print("\n2️⃣  QUESTION 2: Remove NULL emp_id")
    print("-" * 80)
    df_clean = df.filter(col("emp_id").isNotNull())
    df_clean.show()
    
    print("\n3️⃣  QUESTION 3: Remove Duplicates")
    print("-" * 80)
    df_dedup = df_clean.dropDuplicates(["emp_id"])
    df_dedup.show()
    
    print("\n4️⃣  QUESTION 4: Second Highest Salary per Department")
    print("-" * 80)
    window_spec = Window.partitionBy("department").orderBy(desc("salary"))
    df_ranked = df_dedup.withColumn("rank", dense_rank().over(window_spec))
    df_ranked.filter(col("rank") == 2).show()
    
    print("\n5️⃣  QUESTION 5: Drop Column and Add New Column")
    print("-" * 80)
    df_transform = df_dedup.drop("department") \
        .withColumn(
            "salary_level",
            when(col("salary") >= 90000, "High")
            .when(col("salary") >= 70000, "Medium")
            .otherwise("Low")
        )
    df_transform.show()
    
    print("\n6️⃣  QUESTION 6: Broadcast Join")
    print("-" * 80)
    manager_data = [
        ("Sales", "John Smith"),
        ("Engineering", "Alice Johnson"),
        ("HR", "Bob Williams"),
        ("Finance", "Carol Davis")
    ]
    manager_schema = StructType([
        StructField("dept", StringType(), True),
        StructField("manager_name", StringType(), True)
    ])
    df_managers = spark.createDataFrame(manager_data, schema=manager_schema)
    
    df_with_mgr = df_dedup.join(
        broadcast(df_managers),
        df_dedup.department == df_managers.dept,
        "left"
    ).drop("dept")
    df_with_mgr.show()
    
    print("\n7️⃣  QUESTION 7: Write into 5 Output Files")
    print("-" * 80)
    print("Code: df_final.repartition(5).write.mode('overwrite').csv(output_path)")
    print("Creates 5 partition files: part-00000, part-00001, ..., part-00004")


# ============================================================================
# Quick Reference & Best Practices
# ============================================================================

QUICK_REFERENCE = """
================================================================================
PYSPARK QUICK REFERENCE - Employee Data Operations
================================================================================

1. READ CSV WITH SCHEMA:
   ─────────────────────
   schema = StructType([
       StructField("emp_name", StringType(), True),
       StructField("emp_id", IntegerType(), True),
       StructField("department", StringType(), True),
       StructField("salary", DoubleType(), True)
   ])
   
   df = spark.read.schema(schema).option("header", "false").csv("path/to/file.csv")

2. FILTER NULL VALUES:
   ───────────────────
   df.filter(col("emp_id").isNotNull())
   df.dropna(subset=["emp_id"])

3. REMOVE DUPLICATES:
   ──────────────────
   df.dropDuplicates(["emp_id"])  # Keep first occurrence
   
   # Alternative with window function:
   window_spec = Window.partitionBy("emp_id").orderBy("emp_name")
   df.withColumn("rn", row_number().over(window_spec)) \
       .filter(col("rn") == 1).drop("rn")

4. SECOND HIGHEST SALARY PER DEPARTMENT:
   ────────────────────────────────────
   window_spec = Window.partitionBy("department").orderBy(desc("salary"))
   df.withColumn("rank", dense_rank().over(window_spec)) \
       .filter(col("rank") == 2)

5. DROP & ADD COLUMNS:
   ───────────────────
   df.drop("department") \
       .withColumn("salary_category", 
                   when(col("salary") >= 100000, "Senior").otherwise("Junior"))

6. BROADCAST JOIN:
   ───────────────
   df.join(broadcast(df_small), join_condition, "left")
   
   # Broadcast small table to all worker nodes for efficiency

7. WRITE TO 5 FILES:
   ────────────────
   df.repartition(5).write.mode("overwrite").csv("output_path")
   
   # Creates: part-00000, part-00001, ..., part-00004

================================================================================
PERFORMANCE TIPS
================================================================================

1. Caching:
   df.cache()  # Cache dataframe for reuse
   df.unpersist()  # Remove from cache

2. Broadcast Join:
   - Use when one table is < 2GB
   - Automatically chosen by Spark, or use broadcast() explicitly

3. Partitioning:
   - Use repartition() for output files
   - Use partitionBy() for writing partitioned tables

4. Schema Inference:
   - Avoid inferSchema=True for large files (slow)
   - Define schema manually

5. Window Functions:
   - Use dense_rank() for ranking with ties
   - Use row_number() for sequential numbering

================================================================================
HDFS vs ADLS Path Examples
================================================================================

HDFS:
  hdfs://namenode:8020/user/data/employees.csv
  hdfs:///user/data/employees.csv

ADLS Gen 2:
  abfss://container@storageaccount.dfs.core.windows.net/path/to/file.csv

ADLS Gen 1:
  adl://storageaccount.azuredatalakestore.net/path/to/file.csv

Local File System:
  file:///path/to/file.csv

================================================================================
COMMON SCENARIOS
================================================================================

Scenario 1: Handle Missing Values
  - isNull() / isNotNull()
  - dropna()
  - fillna()

Scenario 2: Data Validation
  - Filter invalid records
  - Log errors
  - Write to quarantine table

Scenario 3: De-duplication
  - dropDuplicates()
  - Window functions with row_number()
  - Group by + aggregate

Scenario 4: Aggregations
  - groupBy().agg()
  - Window functions
  - SQL queries

Scenario 5: Data Quality
  - Check null counts
  - Validate constraints
  - Compare before/after counts
"""


if __name__ == "__main__":
    print(QUICK_REFERENCE)
    
    # Run end-to-end pipeline
    end_to_end_pipeline()
    
    # Stop Spark session
    # spark.stop()
