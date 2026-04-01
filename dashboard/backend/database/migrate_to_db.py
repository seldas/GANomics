import os
import yaml
from database import SessionLocal, Project, ProjectConfig, ProjectFile

def migrate_projects():
    session = SessionLocal()
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')

    for project_id in os.listdir(dataset_dir):
        project_dir = os.path.join(dataset_dir, project_id)
        if not os.path.isdir(project_dir): continue

        config_files = [f for f in os.listdir(project_dir) if f.endswith("_config.yaml")]
        for config_file in config_files:
            full_path = os.path.join(project_dir, config_file)
            try:
                with open(full_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            except:
                continue

            metadata = config_data.get('metadata', {})
            project = Project(
                id=project_id,
                name=metadata.get('name', project_id),
                description=metadata.get('description', ""),
                genes=metadata.get('genes'),
                samples=metadata.get('samples'),
                has_label=os.path.exists(os.path.join(project_dir, "label.txt")),
                config_path=full_path
            )
            session.add(project)

            project_config = ProjectConfig(
                project_id=project_id,
                config=config_data
            )
            session.add(project_config)

            project_file = ProjectFile(
                project_id=project_id,
                file_type='config',
                file_path=full_path
            )
            session.add(project_file)

            session.commit()
            break

if __name__ == "__main__":
    migrate_projects()