from sqlalchemy import create_engine, Column, Integer, String, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BACKEND_DIR, "database", "ganomics.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"

    dataset_name = Column(String, primary_key=True, index=True)
    folder = Column(String)
    config_file = Column(String)

    # Optional cached metadata for faster UI display
    description = Column(String, nullable=True)
    genes = Column(Integer, nullable=True)
    samples = Column(Integer, nullable=True)
    has_label = Column(Boolean, default=False)

    experiments = relationship(
        "Experiment",
        back_populates="dataset",
        cascade="all, delete-orphan"
    )


class Experiment(Base):
    __tablename__ = "experiments"

    exp_name = Column(String, primary_key=True, index=True)
    dataset_name = Column(String, ForeignKey("datasets.dataset_name"))
    result_category = Column(String, index=True)

    # Existing path-based fields
    training_checkpoints_folder = Column(String, nullable=True)
    training_logs = Column(String, nullable=True)
    sync_data_files = Column(JSON, nullable=True)
    comparative_analysis_results = Column(String, nullable=True)
    deg_analysis_result_folder = Column(String, nullable=True)
    pathway_result_folder = Column(String, nullable=True)
    modeling_result_folder = Column(String, nullable=True)

    # Cached summary/status fields for fast API responses
    training_status = Column(String, nullable=True)  # idle, running, completed
    has_sync = Column(Boolean, default=False)
    has_comparative = Column(Boolean, default=False)
    has_deg = Column(Boolean, default=False)
    has_pathway = Column(Boolean, default=False)
    has_prediction = Column(Boolean, default=False)

    # Parsed/cached metadata
    sample_count = Column(Integer, nullable=True)
    gene_count = Column(Integer, nullable=True)
    mtime = Column(Integer, nullable=True)  # unix timestamp, easier for frontend

    # Optional grouping/display helpers
    major_group = Column(Integer, nullable=True)
    size = Column(Integer, nullable=True)
    repeats = Column(Integer, nullable=True)

    # Cached algorithm-level status/details
    comparative_algorithms = Column(JSON, nullable=True)
    deg_algorithms = Column(JSON, nullable=True)
    pathway_algorithms = Column(JSON, nullable=True)
    prediction_algorithms = Column(JSON, nullable=True)

    # External test set cached summaries
    ext_ids = Column(JSON, nullable=True)
    ext_statuses = Column(JSON, nullable=True)

    dataset = relationship("Dataset", back_populates="experiments")


Base.metadata.create_all(bind=engine)