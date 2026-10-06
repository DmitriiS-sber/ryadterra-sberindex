"""Bundle source, licensed data, saved results and final submission materials."""

from pathlib import Path
import hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    excluded_prefixes = [
        ".codex-finalizer/",
        "presentation/build/",
        "presentation/preview/",
        "report/preview/",
        "results/smoke/",
        "results/config_smoke/",
    ]
    raw_allowed = {
        "data/raw/consumption.parquet",
        "data/raw/data_license.pdf",
        "data/raw/data_license.txt",
    }
    slide_allowed = {
        "presentation/output/SberIndex_Task2_Submission.pptx",
        "presentation/output/SberIndex_Task2_Submission.pdf",
    }
    files = []
    allowed_roots = {
        "configs",
        "data",
        "docs",
        "presentation",
        "report",
        "results",
        "src",
    }
    allowed_root_files = {
        "README.md",
        "requirements.txt",
        "requirements-core.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "Makefile",
        "LICENSE",
        ".gitignore",
    }
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT).as_posix()
        if (
            path.parts[len(ROOT.parts)] not in allowed_roots
            and relative not in allowed_root_files
        ):
            continue
        if (
            not path.is_file()
            or "__pycache__" in relative
            or relative.endswith(".log")
            or relative == "results/deliverable_manifest.json"
        ):
            continue
        if any(relative.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if relative.startswith("data/raw/") and relative not in raw_allowed:
            continue
        if (
            relative.startswith("presentation/output/")
            and relative not in slide_allowed
        ):
            continue
        files.append((relative, path))
    names = {x[0] for x in files}
    mandatory = {
        "README.md",
        "requirements.txt",
        "configs/experiment.yaml",
        "src/forecast.py",
        "src/detect.py",
        "src/future_warning.py",
        "report/SberIndex_Task2_Report.pdf",
        *slide_allowed,
        "results/verification.json",
        "results/default_baseline/metrics_comparison.csv",
    }
    assert mandatory <= names, mandatory - names
    manifest = {
        "created": "2026-10-05",
        "archive_scope": "Original code, SberIndex CC BY-SA data, results and final report/slides. No environment or model weights.",
        "files": [
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in files
        ],
    }
    mp = ROOT / "results/deliverable_manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    files.append(("results/deliverable_manifest.json", mp))
    archive = ROOT.parent / "SberIndex_Task2_Project.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        for name, path in files:
            z.write(path, "sberindex_task2/" + name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in manifest["files"]:
            assert (
                hashlib.sha256(z.read("sberindex_task2/" + r["path"])).hexdigest()
                == r["sha256"]
            )
    print(
        json.dumps(
            {
                "path": str(archive),
                "files": len(files),
                "bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
