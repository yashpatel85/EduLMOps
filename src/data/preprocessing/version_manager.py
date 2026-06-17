from pathlib import Path


PROCESSED_DIR = Path(
    "data/processed"
)


def get_next_version():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_versions = [
        p.name
        for p in PROCESSED_DIR.iterdir()
        if p.is_dir()
    ]

    if not existing_versions:
        return "educational_qa_v1"

    version_numbers = []

    for version in existing_versions:

        try:
            version_numbers.append(
                int(version.split("_v")[-1])
            )

        except:
            pass

    next_version = max(version_numbers) + 1

    return f"educational_qa_v{next_version}"