from pathlib import Path

import mlflow
from lightning import seed_everything

if __name__ == '__main__':
    seed_everything(seed=42, workers=True)

    root_dir = Path(__file__).parents[2]

    with mlflow.start_run(experiment_id='3', run_name='v1'):
        mlflow.log_artifact(str(root_dir / 'dataset'))
