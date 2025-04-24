# db.py

import os
from sqlalchemy import (
    create_engine, Column, Integer, String,
    LargeBinary, ForeignKey
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

# ─── Pull in your Render DATABASE_URL ─────────────────────────────────────────
# Make sure in your Render dashboard you've set:
#   DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"sslmode": "require"},  # force SSL
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


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
    images = relationship(
        "GestureImage",
        back_populates="gesture",
        cascade="all, delete-orphan",
    )


class GestureImage(Base):
    __tablename__ = "gesture_images"
    id         = Column(Integer, primary_key=True, index=True)
    gesture_id = Column(Integer, ForeignKey("gestures.id"), nullable=False)
    filename   = Column(String, nullable=False)
    image      = Column(LargeBinary, nullable=False)
    gesture    = relationship("Gesture", back_populates="images")
