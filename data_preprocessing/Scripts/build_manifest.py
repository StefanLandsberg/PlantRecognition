import argparse
import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from urllib.parse import urlparse


try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None


def validate_and_clean(df, invasive_species):

    required_cols = ["id", "species_guess", "latitude", "longitude", "image_url", "observed_on"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input CSV")
    df = df.dropna(subset=["species_guess", "latitude", "longitude", "image_url"])
    df = df[
        (df["latitude"].between(-35, -22, inclusive="both"))
        & (df["longitude"].between(16, 33, inclusive="both"))
    ]

    if pd.api.types.is_string_dtype(df["observed_on"]):
        df["observed_on"] = pd.to_datetime(df["observed_on"], errors="coerce")
    df = df.dropna(subset=["observed_on"])

    df["is_invasive"] = df["species_guess"].str.lower().isin(
        [s.lower() for s in invasive_species]
    )

    manifest = pd.DataFrame({
        "observation_id": df["id"].astype(str),
        "species": df["species_guess"].astype(str).str.strip(),
        "image_url": df["image_url"].astype(str).apply(lambda x: x.strip()),
        "lat": df["latitude"].astype(float),
        "lng": df["longitude"].astype(float),
        "observed_on": df["observed_on"],
        "is_invasive": df["is_invasive"].astype(bool),
        "source": "csv"
    })

    return manifest


def write_parquet(df, output_path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    print(f"[✓] Parquet manifest written to {output_path} ({len(df)} rows)")


def migrate_to_mongo(df):
    if MongoClient is None:
        print("[!] pymongo not installed; skipping migration.")
        return

    mongo_uri = os.environ.get("MONGODB_URI")
    if not mongo_uri:
        print("[!] MONGODB_URI not set; skipping migration.")
        return

    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    coll = db.get_collection("plants_manifest")

    ops = []
    for rec in df.to_dict(orient="records"):
        ops.append(
            {
                "update_one": {
                    "filter": {"observation_id": rec["observation_id"]},
                    "update": {"$set": rec},
                    "upsert": True
                }
            }
        )

    if ops:
        res = coll.bulk_write(ops)
        print(f"[✓] MongoDB upsert complete: {res.upserted_count} inserted, {res.modified_count} modified.")
    else:
        print("[!] No records to migrate.")

    client.close()


def main():
    parser = argparse.ArgumentParser(description="Build Parquet manifest for plant observations.")
    parser.add_argument("--input", required=True, help="CSV file with raw observations")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--invasive_list", required=True, help="Text file listing invasive species")
    parser.add_argument("--migrate", action="store_true", help="Also push manifest to MongoDB")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[x] Input CSV not found: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.invasive_list):
        print(f"[x] Invasive list file not found: {args.invasive_list}")
        sys.exit(1)

    df = pd.read_csv(args.input)
    with open(args.invasive_list, encoding="utf-8") as f:
        invasive_species = [line.strip() for line in f if line.strip()]

    manifest = validate_and_clean(df, invasive_species)
    write_parquet(manifest, args.output)

    if args.migrate:
        migrate_to_mongo(manifest)


if __name__ == "__main__":
    main()
'''
manifest.parquet
/PlantRecognition
python data_preprocessing/Scripts/build_manifest.py \
  --input data_preprocessing/data/observations-582302.csv \
  --output data_preprocessing/data/manifest.parquet \
  --invasive_list data_preprocessing/data/invasive_species.txt
  '''
