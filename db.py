# db.py

import os
from sqlalchemy import (
    create_engine, Column, Integer, String,
    LargeBinary, ForeignKey
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL")
engine        = create_engine(DATABASE_URL, echo=False)
SessionLocal  = sessionmaker(bind=engine)
Base          = declarative_base()

class Face(Base):
    __tablename__ = "faces"
    id       = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    image    = Column(LargeBinary, nullable=False)

class Gesture(Base):
    __tablename__ = "gestures"
    id     = Column(Integer, primary_key=True, index=True)
    name   = Column(String, unique=True, nullable=False)
    key    = Column(String, nullable=False)
    images = relationship("GestureImage", back_populates="gesture",
                          cascade="all, delete-orphan")

class GestureImage(Base):
    __tablename__ = "gesture_images"
    id         = Column(Integer, primary_key=True, index=True)
    gesture_id = Column(Integer, ForeignKey("gestures.id"), nullable=False)
    filename   = Column(String, nullable=False)
    image      = Column(LargeBinary, nullable=False)
    gesture    = relationship("Gesture", back_populates="images")
