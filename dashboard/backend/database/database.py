from sqlalchemy import create_engine, Column, Integer, String, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BACKEND_DIR, 'database', 'ganomics.db')
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"

    dataset_name = Column(String, primary_key=True, index=True)
    folder = Column(String)
    config_file = Column(String)

    experiments = relationship("Experiment", back_populates="dataset")

class Experiment(Base):
    __tablename__ = "experiments"

    exp_name = Column(String, primary_key=True, index=True)
    dataset_name = Column(String, ForeignKey("datasets.dataset_name"))
    result_category = Column(String)
    training_checkpoints_folder = Column(String)
    training_logs = Column(String)
    sync_data_files = Column(JSON)
    comparative_analysis_results = Column(String)
    deg_analysis_result_folder = Column(String)
    modeling_result_folder = Column(String)

    dataset = relationship("Dataset", back_populates="experiments")

Base.metadata.create_all(bind=engine)