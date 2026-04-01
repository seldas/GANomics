import os
import re
import json
from typing import Optional, Dict, Any, List

import pandas as pd
import yaml

from database import SessionLocal, Dataset, Experiment


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def existing_file(path: str) -> Optional[str]:
    return path if os.path.isfile(path) else None


def existing_dir(path: str) -> Optional[str]:
    return path if os.path.isdir(path) else None


def safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path) if os.path.isdir(path) else []
    except Exception:
        return []


def read_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def count_table_shape(path: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if not path or not os.path.isfile(path):
        return None, None
    try:
        df = pd.read_csv(path, sep="\t", index_col=0)
        return len(df), len(df.columns)
    except Exception:
        return None, None


def count_columns_only(path: Optional[str]) -> Optional[int]:
    if not path or not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path, sep=None, engine="python", index_col=0, nrows=0)
        return len(df.columns)
    except Exception:
        return None


def count_rows_fast(path: Optional[str]) -> Optional[int]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return None


def build_sync_data_files(sync_data_dir: str, exp_name: str) -> Dict[str, Optional[str]]:
    candidates = {
        "train_microarray_real": os.path.join(sync_data_dir, exp_name, "train", "microarray_real.csv"),
        "train_microarray_fake": os.path.join(sync_data_dir, exp_name, "train", "microarray_fake.csv"),
        "test_microarray_real": os.path.join(sync_data_dir, exp_name, "test", "microarray_real.csv"),
        "test_microarray_fake": os.path.join(sync_data_dir, exp_name, "test", "microarray_fake.csv"),
        "train_rnaseq_real": os.path.join(sync_data_dir, exp_name, "train", "rnaseq_real.csv"),
        "train_rnaseq_fake": os.path.join(sync_data_dir, exp_name, "train", "rnaseq_fake.csv"),
        "test_rnaseq_real": os.path.join(sync_data_dir, exp_name, "test", "rnaseq_real.csv"),
        "test_rnaseq_fake": os.path.join(sync_data_dir, exp_name, "test", "rnaseq_fake.csv"),
    }
    return {k: existing_file(v) for k, v in candidates.items()}


def has_any_real_value(d: Optional[dict]) -> bool:
    return isinstance(d, dict) and any(bool(v) for v in d.values())


def parse_size_and_repeats(exp_name: str) -> tuple[Optional[int], Optional[int]]:
    size = None
    repeats = None
    try:
        if "NB_Size_" in exp_name:
            parts = exp_name.split("_")
            size = int(parts[2])
            repeats = int(parts[4])
        elif "CycleGAN_" in exp_name:
            parts = exp_name.split("_")
            if len(parts) >= 3:
                size = int(parts[1])
                repeats = int(parts[2])
        else:
            parts = exp_name.split("_")
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                size = int(parts[1])
                repeats = int(parts[2])
    except Exception:
        pass
    return size, repeats


def classify_major_group(dataset_name: str, exp_name: str, result_category: Optional[str]) -> int:
    dataset_lower = (dataset_name or "").lower()
    exp_lower = (exp_name or "").lower()
    category_lower = (result_category or "").lower()

    if (
        dataset_name in ["NB", "CycleGAN"]
        or "neuroblastoma" in dataset_lower
        or "nb" in dataset_lower
        or "cyclegan" in dataset_lower
        or "nb" in exp_lower
        or "cyclegan" in exp_lower
        or "nb" in category_lower
        or "cyclegan" in category_lower
    ):
        return 0
    return 1


def infer_training_status(training_checkpoints_folder: Optional[str], training_logs: Optional[str]) -> str:
    if training_checkpoints_folder:
        latest_ckpt = os.path.join(training_checkpoints_folder, "net_latest.pth")
        if os.path.isfile(latest_ckpt):
            return "completed"
    if training_logs and os.path.isfile(training_logs):
        return "idle"
    return "idle"


def parse_comparative_algorithms(comparative_csv: Optional[str]) -> Optional[Dict[str, bool]]:
    if not comparative_csv or not os.path.isfile(comparative_csv):
        return None

    algos = ["GANomics", "COMBAT", "YUGENE", "CUBLOCK", "TDM", "QN"]
    status = {a: False for a in algos}

    try:
        df = pd.read_csv(comparative_csv)
        if "Algorithm" not in df.columns:
            return status

        values = df["Algorithm"].astype(str).str.upper().tolist()
        for algo in algos:
            search_term = "QUANTILE" if algo == "QN" else algo.upper()
            status[algo] = any(search_term in v for v in values)
        return status
    except Exception:
        return status


def parse_deg_algorithms(deg_dir: Optional[str]) -> Optional[Dict[str, bool]]:
    if not deg_dir or not os.path.isdir(deg_dir):
        return None

    algos = ["GANomics", "COMBAT", "YUGENE", "CUBLOCK", "TDM", "QN"]
    files = safe_listdir(deg_dir)
    status = {}
    for algo in algos:
        status[algo] = any(
            f.startswith("Jaccard_Curve_") and algo.lower() in f.lower()
            for f in files
        )
    return status


def parse_prediction_algorithms(pred_dir: Optional[str]) -> Optional[Dict[str, bool]]:
    if not pred_dir or not os.path.isdir(pred_dir):
        return None

    algos = ["GANomics", "COMBAT", "YUGENE", "CUBLOCK", "TDM", "QN"]
    files = safe_listdir(pred_dir)
    status = {}
    for algo in algos:
        status[algo] = any(
            f.startswith("Classifier_Performance_") and algo.lower() in f.lower()
            for f in files
        )
    return status


def parse_pathway_algorithms(pathway_dir: Optional[str]) -> Optional[Dict[str, bool]]:
    if not pathway_dir or not os.path.isdir(pathway_dir):
        return None

    algos = ["GANomics", "COMBAT", "YUGENE", "CUBLOCK", "TDM", "QN"]
    files = [f for f in safe_listdir(pathway_dir) if f.endswith(".csv")]
    status = {a: False for a in algos}

    for f in files:
        lower = f.lower()
        for algo in algos:
            if algo.lower() in lower:
                status[algo] = True

    return status


def infer_pathway_result_folder(biomarkers_dir: str, exp_name: str) -> Optional[str]:
    candidates = [
        os.path.join(biomarkers_dir, "Pathway", exp_name),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def build_internal_metadata(sync_data_files: Dict[str, Optional[str]]) -> tuple[Optional[int], Optional[int]]:
    genes = None
    samples = None

    fake_path = sync_data_files.get("test_microarray_fake")
    real_path = sync_data_files.get("test_microarray_real")

    genes = count_columns_only(fake_path) or count_columns_only(real_path)
    samples = count_rows_fast(real_path) or count_rows_fast(fake_path)

    return samples, genes


def collect_external_statuses(dataset_folder: str, exp_name: str, sync_data_dir: str) -> tuple[List[str], Dict[str, Any]]:
    ext_ids: List[str] = []
    ext_statuses: Dict[str, Any] = {}

    for item in safe_listdir(dataset_folder):
        ext_dir = os.path.join(dataset_folder, item)
        if not (item.startswith("ext_") and os.path.isdir(ext_dir)):
            continue

        ext_ids.append(item)

        meta = {"description": "External Testing dataset", "samples": 0, "genes": 0}
        meta_path = os.path.join(ext_dir, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                pass

        result_dir = os.path.join(sync_data_dir, exp_name, item)
        pathway_dir = os.path.join(result_dir, "Pathway")
        prediction_dir = os.path.join(result_dir, "Prediction")
        deg_dir = os.path.join(result_dir, "DEG")

        ext_statuses[item] = {
            "metadata": meta,
            "sync": (
                os.path.isfile(os.path.join(result_dir, "microarray_fake.csv"))
                or os.path.isfile(os.path.join(result_dir, "translated_ag.tsv"))
                or os.path.isfile(os.path.join(result_dir, "translated_rs.tsv"))
            ),
            "comparative": os.path.isfile(os.path.join(result_dir, "Test_performance.csv")),
            "deg": any(f.startswith("Jaccard_Curve_") for f in safe_listdir(deg_dir)),
            "pathway": any(f.endswith(".csv") for f in safe_listdir(pathway_dir)),
            "pred_model": any(f.startswith("Classifier_Performance_") for f in safe_listdir(prediction_dir)),
        }

    return ext_ids, ext_statuses


def get_dataset_metadata(dataset_folder: str, config_file: str) -> tuple[Optional[str], Optional[int], Optional[int], bool]:
    description = None
    genes = None
    samples = None
    has_label = os.path.isfile(os.path.join(dataset_folder, "label.txt"))

    cfg = read_yaml(config_file)
    meta = cfg.get("metadata", {}) if isinstance(cfg, dict) else {}

    description = meta.get("description")
    genes = meta.get("genes")
    samples = meta.get("samples")

    if genes is None or samples is None:
        df_ag_path = os.path.join(dataset_folder, "df_ag.tsv")
        s_count, g_count = count_table_shape(df_ag_path)
        samples = samples if samples is not None else s_count
        genes = genes if genes is not None else g_count

    return description, genes, samples, has_label


def upsert_dataset(
    session,
    dataset_name: str,
    dataset_folder: str,
    config_file: str,
    description: Optional[str],
    genes: Optional[int],
    samples: Optional[int],
    has_label: bool,
) -> None:
    existing_dataset = session.query(Dataset).filter_by(dataset_name=dataset_name).first()

    if existing_dataset:
        existing_dataset.folder = dataset_folder
        existing_dataset.config_file = config_file
        existing_dataset.description = description
        existing_dataset.genes = genes
        existing_dataset.samples = samples
        existing_dataset.has_label = has_label
    else:
        session.add(
            Dataset(
                dataset_name=dataset_name,
                folder=dataset_folder,
                config_file=config_file,
                description=description,
                genes=genes,
                samples=samples,
                has_label=has_label,
            )
        )


def upsert_experiment(
    session,
    exp_name: str,
    dataset_name: str,
    result_category: str,
    training_checkpoints_folder: Optional[str],
    training_logs: Optional[str],
    sync_data_files: Dict[str, Optional[str]],
    comparative_analysis_results: Optional[str],
    deg_analysis_result_folder: Optional[str],
    pathway_result_folder: Optional[str],
    modeling_result_folder: Optional[str],
    training_status: str,
    has_sync: bool,
    has_comparative: bool,
    has_deg: bool,
    has_pathway: bool,
    has_prediction: bool,
    sample_count: Optional[int],
    gene_count: Optional[int],
    mtime: Optional[int],
    major_group: Optional[int],
    size: Optional[int],
    repeats: Optional[int],
    comparative_algorithms: Optional[Dict[str, bool]],
    deg_algorithms: Optional[Dict[str, bool]],
    pathway_algorithms: Optional[Dict[str, bool]],
    prediction_algorithms: Optional[Dict[str, bool]],
    ext_ids: Optional[List[str]],
    ext_statuses: Optional[Dict[str, Any]],
) -> None:
    existing_experiment = session.query(Experiment).filter_by(exp_name=exp_name).first()

    payload = dict(
        dataset_name=dataset_name,
        result_category=result_category,
        training_checkpoints_folder=training_checkpoints_folder,
        training_logs=training_logs,
        sync_data_files=sync_data_files,
        comparative_analysis_results=comparative_analysis_results,
        deg_analysis_result_folder=deg_analysis_result_folder,
        pathway_result_folder=pathway_result_folder,
        modeling_result_folder=modeling_result_folder,
        training_status=training_status,
        has_sync=has_sync,
        has_comparative=has_comparative,
        has_deg=has_deg,
        has_pathway=has_pathway,
        has_prediction=has_prediction,
        sample_count=sample_count,
        gene_count=gene_count,
        mtime=mtime,
        major_group=major_group,
        size=size,
        repeats=repeats,
        comparative_algorithms=comparative_algorithms,
        deg_algorithms=deg_algorithms,
        pathway_algorithms=pathway_algorithms,
        prediction_algorithms=prediction_algorithms,
        ext_ids=ext_ids,
        ext_statuses=ext_statuses,
    )

    if existing_experiment:
        for key, value in payload.items():
            setattr(existing_experiment, key, value)
    else:
        session.add(Experiment(exp_name=exp_name, **payload))


def collect_experiments_for_category(dataset_name: str, dataset_folder: str, result_category: str) -> List[Dict[str, Any]]:
    results_dir = os.path.join(BASE_DIR, result_category)
    if not os.path.isdir(results_dir):
        return []

    training_dir = os.path.join(results_dir, "1_Training")
    logs_dir = os.path.join(training_dir, "logs")
    checkpoints_dir = os.path.join(training_dir, "checkpoints")
    sync_data_dir = os.path.join(results_dir, "2_SyncData")
    comparative_dir = os.path.join(results_dir, "3_ComparativeAnalysis")
    biomarkers_dir = os.path.join(results_dir, "4_Biomarkers")

    if not os.path.isdir(logs_dir):
        return []

    experiments = []

    for filename in os.listdir(logs_dir):
        if not filename.endswith(".txt"):
            continue

        exp_name = re.sub(r"_log\.txt$", "", filename)
        exp_name = re.sub(r"\.txt$", "", exp_name)

        if not exp_name.startswith(dataset_name):
            continue

        log_candidates = [
            os.path.join(logs_dir, f"{exp_name}_log.txt"),
            os.path.join(logs_dir, f"{exp_name}.txt"),
        ]
        training_logs = next((p for p in log_candidates if os.path.isfile(p)), None)
        training_checkpoints_folder = existing_dir(os.path.join(checkpoints_dir, exp_name))
        sync_data_files = build_sync_data_files(sync_data_dir, exp_name)
        comparative_analysis_results = existing_file(
            os.path.join(comparative_dir, exp_name, "Test_performance.csv")
        )
        deg_analysis_result_folder = existing_dir(
            os.path.join(biomarkers_dir, "DEG", exp_name)
        )
        pathway_result_folder = infer_pathway_result_folder(biomarkers_dir, exp_name)
        modeling_result_folder = existing_dir(
            os.path.join(biomarkers_dir, "Prediction", exp_name)
        )

        training_status = infer_training_status(training_checkpoints_folder, training_logs)
        has_sync = has_any_real_value(sync_data_files)
        has_comparative = comparative_analysis_results is not None
        has_deg = deg_analysis_result_folder is not None and any(
            f.endswith(".csv") for f in safe_listdir(deg_analysis_result_folder)
        )
        has_pathway = pathway_result_folder is not None and any(
            f.endswith(".csv") for f in safe_listdir(pathway_result_folder)
        )
        has_prediction = modeling_result_folder is not None and any(
            f.endswith(".csv") for f in safe_listdir(modeling_result_folder)
        )

        sample_count, gene_count = build_internal_metadata(sync_data_files)

        mtime = None
        if training_logs and os.path.isfile(training_logs):
            try:
                mtime = int(os.path.getmtime(training_logs))
            except Exception:
                pass

        size, repeats = parse_size_and_repeats(exp_name)
        major_group = classify_major_group(dataset_name, exp_name, result_category)

        comparative_algorithms = parse_comparative_algorithms(comparative_analysis_results)
        deg_algorithms = parse_deg_algorithms(deg_analysis_result_folder)
        pathway_algorithms = parse_pathway_algorithms(pathway_result_folder)
        prediction_algorithms = parse_prediction_algorithms(modeling_result_folder)

        ext_ids, ext_statuses = collect_external_statuses(dataset_folder, exp_name, sync_data_dir)

        experiments.append(
            {
                "exp_name": exp_name,
                "dataset_name": dataset_name,
                "result_category": result_category,
                "training_checkpoints_folder": training_checkpoints_folder,
                "training_logs": training_logs,
                "sync_data_files": sync_data_files,
                "comparative_analysis_results": comparative_analysis_results,
                "deg_analysis_result_folder": deg_analysis_result_folder,
                "pathway_result_folder": pathway_result_folder,
                "modeling_result_folder": modeling_result_folder,
                "training_status": training_status,
                "has_sync": has_sync,
                "has_comparative": has_comparative,
                "has_deg": has_deg,
                "has_pathway": has_pathway,
                "has_prediction": has_prediction,
                "sample_count": sample_count,
                "gene_count": gene_count,
                "mtime": mtime,
                "major_group": major_group,
                "size": size,
                "repeats": repeats,
                "comparative_algorithms": comparative_algorithms,
                "deg_algorithms": deg_algorithms,
                "pathway_algorithms": pathway_algorithms,
                "prediction_algorithms": prediction_algorithms,
                "ext_ids": ext_ids,
                "ext_statuses": ext_statuses,
            }
        )

    return experiments


def migrate_datasets_and_experiments() -> None:
    session = SessionLocal()

    try:
        if not os.path.isdir(DATASET_DIR):
            raise RuntimeError(f"Dataset directory not found: {DATASET_DIR}")

        for dataset_name in os.listdir(DATASET_DIR):
            dataset_folder = os.path.join(DATASET_DIR, dataset_name)
            if not os.path.isdir(dataset_folder):
                continue

            config_files = [f for f in os.listdir(dataset_folder) if f.endswith("_config.yaml")]
            if not config_files:
                print(f"[SKIP] No config file found for dataset: {dataset_name}")
                continue

            full_config_path = os.path.join(dataset_folder, config_files[0])
            description, genes, samples, has_label = get_dataset_metadata(dataset_folder, full_config_path)

            upsert_dataset(
                session=session,
                dataset_name=dataset_name,
                dataset_folder=dataset_folder,
                config_file=full_config_path,
                description=description,
                genes=genes,
                samples=samples,
                has_label=has_label,
            )

            for result_category in ["results", "results_ms"]:
                experiments = collect_experiments_for_category(dataset_name, dataset_folder, result_category)

                for exp in experiments:
                    upsert_experiment(
                        session=session,
                        exp_name=exp["exp_name"],
                        dataset_name=exp["dataset_name"],
                        result_category=exp["result_category"],
                        training_checkpoints_folder=exp["training_checkpoints_folder"],
                        training_logs=exp["training_logs"],
                        sync_data_files=exp["sync_data_files"],
                        comparative_analysis_results=exp["comparative_analysis_results"],
                        deg_analysis_result_folder=exp["deg_analysis_result_folder"],
                        pathway_result_folder=exp["pathway_result_folder"],
                        modeling_result_folder=exp["modeling_result_folder"],
                        training_status=exp["training_status"],
                        has_sync=exp["has_sync"],
                        has_comparative=exp["has_comparative"],
                        has_deg=exp["has_deg"],
                        has_pathway=exp["has_pathway"],
                        has_prediction=exp["has_prediction"],
                        sample_count=exp["sample_count"],
                        gene_count=exp["gene_count"],
                        mtime=exp["mtime"],
                        major_group=exp["major_group"],
                        size=exp["size"],
                        repeats=exp["repeats"],
                        comparative_algorithms=exp["comparative_algorithms"],
                        deg_algorithms=exp["deg_algorithms"],
                        pathway_algorithms=exp["pathway_algorithms"],
                        prediction_algorithms=exp["prediction_algorithms"],
                        ext_ids=exp["ext_ids"],
                        ext_statuses=exp["ext_statuses"],
                    )

            session.commit()
            print(f"[OK] Migrated dataset: {dataset_name}")

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    migrate_datasets_and_experiments()