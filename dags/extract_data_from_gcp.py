from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
import pandas as pd
import sqlalchemy

def load_to_sql(file_path):
    conn = BaseHook.get_connection('postgres_default')  
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@project-2-titanic-survival-prediction_6ac56e-postgres-1:{conn.port}/{conn.schema}")
    df = pd.read_csv(file_path)
    df.to_sql(name="titanic", con=engine, if_exists="replace", index=False)

# Define the DAG
with DAG(
    dag_id="extract_titanic_data",
    schedule=None, 
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    list_files = GCSListObjectsOperator(
        task_id="list_files",
        bucket="my_bucket_mlops_udemy_project3", 
    )

    download_file = GCSToLocalFilesystemOperator(
        task_id="download_file",
        bucket="my_bucket_mlops_udemy_project3", 
        object_name="TitanicDataset.csv", 
        filename="/tmp/TitanicDataset.csv", 
    )

    load_data = PythonOperator(
        task_id="load_to_sql",
        python_callable=load_to_sql,
        op_kwargs={"file_path": "/tmp/TitanicDataset.csv"}
    )

    list_files >> download_file >> load_data
