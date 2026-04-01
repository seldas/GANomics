from sqlalchemy import create_engine, Column, Integer, String, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os

DATABASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(DATABASE_DIR, 'ganomics.db')
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    genes = Column(Integer)
    samples = Column(Integer)
    has_label = Column(Boolean)
    config_path = Column(String)

    config = relationship("ProjectConfig", back_populates="project")
    files = relationship("ProjectFile", back_populates="project")

class ProjectConfig(Base):
    __tablename__ = "project_configs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"))
    config = Column(JSON)

    project = relationship("Project", back_populates="config")

class ProjectFile(Base):
    __tablename__ = "project_files"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"))
    file_type = Column(String)
    file_path = Column(String)

    project = relationship("Project", back_populates="files")

Base.metadata.create_all(bind=engine)