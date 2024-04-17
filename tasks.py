# coding: utf-8

import json
import logging
import os
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath

import requests
from huggingface_hub import HfApi
from invoke import task


_LOGGER = logging.getLogger("piper.rt")
PIPER_CHECKPOINTS_DATASET = "rhasspy/piper-checkpoints"
RT_VOICES_DATASET = "mush42/piper-rt"
CHECKPOINTS_URL_PREFIX = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/{}"


@dataclass
class Voice:
    name: str
    config: str
    checkpoint: str
    etag: str


def export_and_package(c, voice, export_script_path, working_dir):
    _LOGGER.info("Downloading checkpoint...")
    export_dir = working_dir.joinpath("exported")
    export_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_url = CHECKPOINTS_URL_PREFIX.format(voice.checkpoint)
    checkpoint_response = requests.get(checkpoint_url, stream=True)
    checkpoint_response.raise_for_status()
    downloaded_checkpoint_filename = working_dir.joinpath("checkpoint.ckpt")
    _LOGGER.info("Downloading checkpoint.")
    with open(downloaded_checkpoint_filename, "wb") as file:
        for chunk in checkpoint_response.iter_content(chunk_size=None):
            file.write(chunk)
    _LOGGER.info("Exporting to ONNX.")
    with c.cd(export_script_path):
        export_cmd = " ".join([
            "python3 -m piper_train.export_onnx_streaming --debug",
            os.fspath(downloaded_checkpoint_filename),
            os.fspath(export_dir),
        ])
        c.run(export_cmd)
    # Config
    _LOGGER.info("Preparing config.")
    config_url = CHECKPOINTS_URL_PREFIX.format(voice.config)
    config_json = requests.get(config_url).json()
    config_json["streaming"] = True
    voice_name_parts = voice.name.split("-")
    new_name = "-".join([
        voice_name_parts[0],
        f"{voice_name_parts[1]}+RT",
        voice_name_parts[2]
    ])
    config_json["key"] = new_name
    export_dir.joinpath(f"{new_name}.json").write_text(
        json.dumps(config_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n"
    )
    _LOGGER.info("Packaging voice...")
    package_filename = working_dir.joinpath(f"{new_name}.tar.gz")
    with tarfile.TarFile(package_filename, "w") as pack:
        for file in export_dir.iterdir():
            pack.add(os.fspath(file.resolve()), arcname=file.name)
    # Upload to hf-hub
    _LOGGER.info("Uploading voice...")
    hf_client.upload_file(
        path_or_fileobj=package_filename,
        path_in_repo=package_filename.name,
        repo_id=RT_VOICES_DATASET,
        repo_type="dataset",
    )
    # Cleanup
    _LOGGER.info("Cleaning up...")
    with c.cd(working_dir):
        c.run("rm -rf *")


def dump_voices_metadata(voices, working_dir):
    _LOGGER.info("Dumping voice metadata.")
    dst_json_filename = working_dir.joinpath("metadata.json")
    data = [
        asdict(voice)
        for voice in voices
    ]
    with open(dst_json_filename, "w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    hf_client.upload_file(
        path_or_fileobj=dst_json_filename,
        path_in_repo=dst_json_filename.name,
        repo_id=RT_VOICES_DATASET,
        repo_type="dataset",
    )


@task
def run(c):
    logging.basicConfig(level=logging.INFO)

    _LOGGER.info("Installing basic deps.")
    c.run("pip3 install -r requirements.txt")
    _LOGGER.info("Cloning piper repo")
    if not Path.cwd().joinpath("piper").is_dir():
        c.run("git clone https://github.com/mush42/piper")
    with c.cd("./piper"):
        c.run("git checkout streaming")
    _LOGGER.info("Installing piper deps.")
    with c.cd("./piper/src/python"):
        c.run("pip3 install -r requirements.txt")
    # Force upgrade torch for best export results
    c.run("pip3 install --upgrade torch")
    # Paths
    export_script_path = Path.cwd().joinpath("piper", "src", "python")
    working_dir = Path.cwd().joinpath("workspace")
    working_dir.mkdir(parents=True, exist_ok=True)
    hf_client = HfApi()
    all_files = hf_client.list_repo_files(
        PIPER_CHECKPOINTS_DATASET,
        repo_type="dataset"
    )
    files = [PurePosixPath(path) for path in all_files]
    config_files = filter(
        lambda f: f.name == 'config.json',
        files
    )
    voices = []
    for cfg_file in config_files:
        voice_name = "-".join(cfg_file.parent.parts[1:])
        try:
            checkpoint_file = next(filter(
                lambda f: (f.parent == cfg_file.parent) and ("epoch" in f.name),
                files
            ))
        except StopIteration:
            continue
        file_metadata = hf_client.get_hf_file_metadata(
            url=CHECKPOINTS_URL_PREFIX.format(checkpoint_file)
        )
        voice = Voice(
            name=voice_name,
            config=os.fspath(cfg_file),
            checkpoint=os.fspath(checkpoint_file),
            etag=file_metadata.etag
        )
        voices.append(voice)
    # Export and package
    for voice in voices:
        _LOGGER.info(f"Processing voice: {voice.name}")
        try:
            export_and_package(c, voice, export_script_path, working_dir)
        except:
            _LOGGER.error("Failed to export and package voice", exc_info=True)
            continue
    # Dump metadata for later reference
    dump_voices_metadata(voices, working_dir)
    _LOGGER.info("Process Done")

