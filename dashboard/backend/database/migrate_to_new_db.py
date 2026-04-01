import os
import yaml
from database import SessionLocal, Dataset, Experiment

def migrate_datasets_and_experiments():
    session = SessionLocal()
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')

    for dataset_name in os.listdir(dataset_dir):
        dataset_folder = os.path.join(dataset_dir, dataset_name)
        if not os.path.isdir(dataset_folder): continue

        config_files = [f for f in os.listdir(dataset_folder) if f.endswith("_config.yaml")]
        for config_file in config_files:
            full_config_path = os.path.join(dataset_folder, config_file)
            dataset = Dataset(
                dataset_name=dataset_name,
                folder=dataset_folder,
                config_file=full_config_path
            )
            session.add(dataset)

            for result_category in ['results', 'results_ms']:
                results_dir = os.path.join(os.path.dirname(__file__), result_category)
                if not os.path.exists(results_dir): continue

                training_dir = os.path.join(results_dir, '1_Training')
                logs_dir = os.path.join(training_dir, 'logs')
                checkpoints_dir = os.path.join(training_dir, 'checkpoints')
                sync_data_dir = os.path.join(results_dir, '2_SyncData')
                comparative_dir = os.path.join(results_dir, '3_ComparativeAnalysis')
                biomarkers_dir = os.path.join(results_dir, '4_Biomarkers')

                for exp_name in os.listdir(logs_dir):
                    if exp_name.startswith(dataset_name):
                        training_logs = os.path.join(logs_dir, exp_name + '_log.txt')
                        training_checkpoints_folder = os.path.join(checkpoints_dir, exp_name)
                        sync_data_files = {
                            'train_microarray_real': os.path.join(sync_data_dir, exp_name, 'train', 'microarray_real.csv'),
                            'train_microarray_fake': os.path.join(sync_data_dir, exp_name, 'train', 'microarray_fake.csv'),
                            'test_microarray_real': os.path.join(sync_data_dir, exp_name, 'test', 'microarray_real.csv'),
                            'test_microarray_fake': os.path.join(sync_data_dir, exp_name, 'test', 'microarray_fake.csv'),
                            'train_rnaseq_real': os.path.join(sync_data_dir, exp_name, 'train', 'rnaseq_real.csv'),
                            'train_rnaseq_fake': os.path.join(sync_data_dir, exp_name, 'train', 'rnaseq_fake.csv'),
                            'test_rnaseq_real': os.path.join(sync_data_dir, exp_name, 'test', 'rnaseq_real.csv'),
                            'test_rnaseq_fake': os.path.join(sync_data_dir, exp_name, 'test', 'rnaseq_fake.csv'),
                        }
                        comparative_analysis_results = os.path.join(comparative_dir, exp_name, 'Test_performance.csv')
                        deg_analysis_result_folder = os.path.join(biomarkers_dir, 'DEG', exp_name)
                        modeling_result_folder = os.path.join(biomarkers_dir, 'Prediction', exp_name)

                        experiment = Experiment(
                            exp_name=exp_name,
                            dataset_name=dataset_name,
                            result_category=result_category,
                            training_checkpoints_folder=training_checkpoints_folder,
                            training_logs=training_logs,
                            sync_data_files=sync_data_files,
                            comparative_analysis_results=comparative_analysis_results,
                            deg_analysis_result_folder=deg_analysis_result_folder,
                            modeling_result_folder=modeling_result_folder
                        )
                        session.add(experiment)

            session.commit()
            break

if __name__ == "__main__":
    migrate_datasets_and_experiments()