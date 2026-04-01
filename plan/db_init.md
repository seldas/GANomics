# Database Initialization Plan

## Overview
This document outlines the steps to transition from file system scanning to using a database in the GANomics API, particularly for the `list_projects` function.

## Steps to Implement Database

1. **Database Setup**:
   - Choose a suitable database system (e.g., PostgreSQL, SQLite).
   - Implement the database schema using SQLAlchemy or another ORM compatible with FastAPI.

2. **Database Schema Design**:
   - Identify the information currently stored in configuration files that needs to be migrated to the database.
   - Design a database schema that can store project metadata (e.g., project ID, name, description, gene count, sample count, configuration details).

3. **Migration Script**:
   - Create a script to migrate existing project information from configuration files to the database.
   - This script should read the configuration files, extract relevant information, and populate the database.

4. **Modify `list_projects` Function**:
   - Update the `list_projects` function to query the database instead of scanning the file system.
   - Ensure it retrieves the necessary project information from the database.

5. **Update Other Relevant Functions**:
   - Identify other parts of the code that involve file system scanning related to project data.
   - Modify these sections to use database queries where appropriate.

6. **Database Session Management**:
   - Implement database session management in the FastAPI application.
   - Use dependency injection to provide database sessions to the route handlers.

7. **Testing**:
   - Thoroughly test the modified `list_projects` function and other affected parts to ensure they work correctly with the database.
   - Verify that project information is accurately retrieved and displayed.

8. **Documentation Update**:
   - Update the API documentation to reflect any changes in the API endpoints or data models.

## Task Progress Checklist
- [ ] Set up the database system
- [ ] Implement the database schema
- [ ] Create a migration script
- [ ] Modify list_projects function
- [ ] Update create_project function
- [ ] Modify other relevant functions
- [ ] Implement database session management
- [ ] Write comprehensive tests
- [ ] Verify data correctness
- [ ] Update API documentation

By following this plan, we can effectively transition from file system scanning to using a database, improving the efficiency and scalability of the project listing functionality.


## update, in database schema
We need to carefully design the database schema to store the information we need to efficiently find the related file and call it.

1. experiment
for each experiment (like NB_Ablation_Size_10_Run_0), it has the following attributes:
exp_name (NB_Ablation_Size_10_Run_0)
dataset (the first split by _, like NB)
result_category (results, results_ms, or results_old)
Training checkpoints folder (e.g., results_ms/1_Training/checkpoints/<exp_name>/, result_ms is the <result_category>) 
Training logs (e.g., results_ms/1_Training/logs/<exp_name>_log.txt)
SyncData, including training and testing, in total 8 file names. All of them should be saved in the db for quick query.
    for example, results_ms/2_SyncData/<exp_name>/train/microarray_fake.csv (test/rnaseq_real.csv, in total 8 files)
comparative analysis results: results_ms/3_ComparativeAnalysis/<exp_name>/Test_performance.csv
DEG analysis result folder: results_ms/4_Biomarkers/DEG/<exp_name>/
Modeling result folder: results_ms/4_Biomarkers/Prediction/<exp_name>/

2. dataset 
dataset_name (NB)
folder: dashboard/backend/dataset/<dataset_name>
config_file (e.g., dashboard/backend/dataset/BRCA-1/brca-1_config.yaml)

## in script
in main.py, I think we don't need to define many of these variables, most of them shouldbe determined by the information in db.


# --- Path Definitions ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", ".."))

DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
RESULTS_DIR = os.path.join(BACKEND_DIR, "results")
SCRIPTS_DIR = os.path.join(BACKEND_DIR, "scripts")
TEMP_DIR = os.path.join(BACKEND_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

TRAINING_DIR = os.path.join(RESULTS_DIR, "1_Training")
LOGS_DIR = os.path.join(TRAINING_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, "checkpoints")

SYNC_DATA_DIR = os.path.join(RESULTS_DIR, "2_SyncData")
COMPARATIVE_DIR = os.path.join(RESULTS_DIR, "3_ComparativeAnalysis")
BIOMARKERS_DIR = os.path.join(RESULTS_DIR, "4_Biomarkers")
FIGURES_DIR = os.path.join(RESULTS_DIR, "5_Figures")
RESULTS_MS_DIR = os.path.join(BACKEND_DIR, "results_ms")
os.makedirs(RESULTS_MS_DIR, exist_ok=True)

MS_TRAINING_DIR = os.path.join(RESULTS_MS_DIR, "1_Training")
MS_LOGS_DIR = os.path.join(MS_TRAINING_DIR, "logs")
MS_SYNC_DIR = os.path.join(RESULTS_MS_DIR, "2_SyncData")
MS_COMPARATIVE_DIR = os.path.join(RESULTS_MS_DIR, "3_ComparativeAnalysis")
MS_BIOMARKERS_DIR = os.path.join(RESULTS_MS_DIR, "4_Biomarkers")
