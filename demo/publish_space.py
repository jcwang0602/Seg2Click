"""Publish the self-contained Seg2Click demo to Hugging Face Spaces."""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="jcwang0602/Seg2Click-Demo")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--static-wrapper", action="store_true", help="Publish the free static redirect wrapper")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    demo_dir = repository_root / "demo"
    with tempfile.TemporaryDirectory(prefix="seg2click-space-") as temporary_dir:
        staging_dir = Path(temporary_dir)
        api = HfApi()
        if args.static_wrapper:
            static_dir = demo_dir / "static"
            shutil.copy2(static_dir / "index.html", staging_dir / "index.html")
            shutil.copy2(static_dir / "README.md", staging_dir / "README.md")
            api.create_repo(
                repo_id=args.repo_id,
                repo_type="space",
                space_sdk="static",
                private=args.private,
                exist_ok=True,
            )
        else:
            shutil.copy2(demo_dir / "app.py", staging_dir / "app.py")
            shutil.copy2(demo_dir / "requirements.txt", staging_dir / "requirements.txt")
            shutil.copy2(demo_dir / "README_SPACE.md", staging_dir / "README.md")
            shutil.copytree(repository_root / "seg2click", staging_dir / "seg2click")
            try:
                api.create_repo(
                    repo_id=args.repo_id,
                    repo_type="space",
                    space_sdk="gradio",
                    private=args.private,
                    exist_ok=True,
                )
            except HfHubHTTPError as error:
                if error.response is None or error.response.status_code != 402:
                    raise
                print("Gradio Space creation requires PRO; publishing a static wrapper for the ZeroGPU demo.")
                for path in staging_dir.iterdir():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                static_dir = demo_dir / "static"
                shutil.copy2(static_dir / "index.html", staging_dir / "index.html")
                shutil.copy2(static_dir / "README.md", staging_dir / "README.md")
                api.create_repo(
                    repo_id=args.repo_id,
                    repo_type="space",
                    space_sdk="static",
                    private=args.private,
                    exist_ok=True,
                )
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=staging_dir,
            commit_message="Deploy Seg2Click interactive demo",
            ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
        )
    print(f"Published https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
