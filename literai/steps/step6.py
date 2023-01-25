import argparse
import json
import os
import sys
from literai.util import get_base_output_dir, get_output_dir, slugify
from literai.steps.util import free_memory_after
from typing import Optional


@free_memory_after
def step6(title: str, author: str, gcloud_credentials: Optional[str], gcloud_bucket: Optional[str]):
    print("------------- STEP 6 (Finalize) ------------- ")

    title_dir = get_output_dir(title)
    parts = [f for f in os.listdir(title_dir) if f.startswith(
        "part") and f.endswith(".json")]

    obj = {
        "title": title,
        "author": author,
        "parts": parts
    }

    title_slug = slugify(title)
    json.dump(obj, open(os.path.join(
        title_dir, f"{title_slug}.json"), "w", encoding="utf-8"), indent=2)

    index_path = os.path.join(get_base_output_dir(), "index.json")

    if gcloud_credentials is not None:
        from google.cloud import storage
        storage_client = storage.Client.from_service_account_json(
            gcloud_credentials)
        storage_bucket = storage_client.bucket(gcloud_bucket)
    else:
        storage_bucket = None

    if storage_bucket is not None:
        index_blob = storage_bucket.blob("index.json")
        if index_blob.exists():
            print(f"Pulling index.json from {gcloud_bucket}")
            index = json.load(index_blob.open("r", encoding="utf-8"))
        else:
            index = []
    elif os.path.exists(index_path):
        index = json.load(open(index_path, "r", encoding="utf-8"))
    else:
        index = []

    obj = {
        "title": title,
        "author": author,
        "data": f"{title_slug}/{title_slug}.json"
    }

    found = False
    for i in range(0, len(index)):
        if slugify(index[i]["title"]) == title_slug:
            index[i] = obj
            found = True
            break
    if not found:
        index.append(obj)

    json.dump(index, storage_bucket.blob("index.json").open("w", encoding="utf-8")
              if storage_bucket is not None else open(index_path, "w", encoding="utf-8"), indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="title of the novel")
    parser.add_argument("author", help="author of the novel")
    parser.add_argument("--gcloud_credentials",
                        help="path to Google Cloud Storage service account .json credentials")
    parser.add_argument("--gcloud_bucket",
                        help="Google Cloud Storage buicket to upload to")
    args = parser.parse_args()

    step6(args.title, args.author, args.gcloud_credentials, args.gcloud_bucket)


if __name__ == '__main__':
    sys.exit(main())
