import os
from sqlalchemy import create_engine, text
ENGINE = create_engine(os.getenv("DATABASE_URL", "sqlite:///resolveai.db"))  # set to postgresql+psycopg2://... on Cloud Run
with ENGINE.begin() as c:
    c.execute(text("CREATE TABLE IF NOT EXISTS assessments (claim_id TEXT, created TEXT, body TEXT)"))
def save(a):
    with ENGINE.begin() as c:
        c.execute(text("INSERT INTO assessments VALUES (:i, CURRENT_TIMESTAMP, :b)"), {"i": a.claim_id, "b": a.model_dump_json()})
