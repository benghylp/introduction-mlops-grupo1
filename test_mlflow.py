import mlflow
from mlflow import log_metric, log_param, log_artifact, start_run

# Tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Indica a MLflow a qué servidor de seguimiento (Tracking Server) debe conectarse para registrar y consultar información sobre los experimentos

# Nombre del Expermiento
mlflow.set_experiment("mi_experimento")

if __name__ == '__main__':
    print("Iniciando ejecución...") 
    with start_run(run_name='run_prueba_02'):
        log_param("threshold", 3)
    
        log_metric("timestamp", 1000)

        log_artifact("produced-dataset.csv")

        mlflow.set_tag('model','prophet')
        mlflow.set_tag('state','dev')
        mlflow.set_tags({
            'author':'Benghy',
            'stage':'development',
            'version':'v2.0'
        })
