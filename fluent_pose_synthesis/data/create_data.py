import os
import math
import json
import logging
import argparse
import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus import DgsCorpusConfig
from pose_format import PoseHeader, Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from spoken_to_signed.gloss_to_pose import concatenate_poses

from fluent_pose_synthesis.data.map_gloss_to_pose import create_gloss_to_pose_dict


# Example call:
# python create_data.py --corpus_dir /scratch/ronli --dictionary_dir /scratch/ronli
# --output_dir /scratch/ronli/output --max_examples 6


@dataclass  # pylint: disable=too-few-public-methods
class ProcessedSentence:
    concatenated_original_pose: Pose
    concatenated_updated_pose: Pose
    metadata: Dict[str, Any]
    valid_gloss_count: int
    replaced_gloss_count: int


class DGSPoseDataset:
    def __init__(
        self, corpus_dir: Path, dictionary_dir: Path, max_examples: Optional[int] = None
    ):
        """
        Initializes the DGSPoseDataset class.
        Args:
            corpus_dir (Path): Path to the directory containing the DGS Corpus dataset.
            dictionary_dir (Path): Path to the directory containing the DGS Types dataset.
            max_examples (Optional[int]): Maximum number of examples to process. If None, process all examples.
        """
        self.max_examples = max_examples
        self.corpus_dir = corpus_dir
        self.dictionary_dir = dictionary_dir
        self.dictionary = self._load_dgs_types_dictionary()

    def _load_dgs_corpus(self) -> tf.data.Dataset:
        """
        Loads the DGS Corpus dataset.
        """
        config_corpus = DgsCorpusConfig(
            name="sentence-level-pose-fps25",
            version="3.0.0",
            include_video=False,
            include_pose="holistic",
            data_type="sentence",
            fps=25,  # Specify FPS as 25 for pose data
            split="3.0.0-uzh-sentence",  # Use the sentence-level split
        )
        dgs_corpus = tfds.load(
            "dgs_corpus",
            builder_kwargs={"config": config_corpus},
            data_dir=self.corpus_dir,
        )

        return dgs_corpus

    def _load_dgs_types_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads the DGS Types dataset and creates a gloss-to-pose dictionary.
        """
        config = SignDatasetConfig(
            name="pose_holistic",
            version="3.0.0",
            include_video=False,
            include_pose="holistic",
            process_pose=False,
        )
        dgs_types = tfds.load(
            "dgs_types",
            builder_kwargs=dict(config=config),
            data_dir=self.dictionary_dir,
        )
        dgs_types_dict, _ = create_gloss_to_pose_dict(dgs_types)

        return dgs_types_dict

    def _should_replace(self, gloss: str) -> bool:
        """
        Determines whether the pose for a given gloss should be replaced.
        Args:
            gloss (str): The gloss to check.
        """
        # Check if a given gloss is in DGS types dictionary and has a corresponding pose,
        # and is not a special generic sign
        return (
            not gloss.startswith("$")
            and gloss in self.dictionary
            and self.dictionary[gloss]["views"]["pose"] is not None
        )

    @lru_cache(maxsize=None)
    def _get_pose_header(self, dataset_name: str) -> PoseHeader:
        """
        Retrieves the pose header for a given dataset.
        Args:
            dataset_name (str): Name of the dataset (e.g., "dgs_corpus").
        """
        dataset_module = importlib.import_module(
            f"sign_language_datasets.datasets.{dataset_name}.{dataset_name}"
        )
        # Read the pose header from the dataset's predefined file
        with open(dataset_module._POSE_HEADERS["holistic"], "rb") as buffer:
            pose_header = PoseHeader.read(BufferReader(buffer.read()))

        return pose_header

    def _create_pose_object(
        self, pose_datum: Dict[str, Any], dataset_name: str
    ) -> Pose:
        """
        Creates a Pose object from the given pose data_entry and dataset name.
        Args:
            pose_datum (Dict[str, Any]): Pose data_entry from the dataset.
            dataset_name (str): Name of the dataset (e.g., "dgs_corpus").
        """
        # Retrieve the pose header for the dataset
        pose_header = self._get_pose_header(dataset_name)
        # Extract pose data_entry
        fps = float(pose_datum["fps"].numpy())
        pose_data = pose_datum["data"].numpy()
        conf = pose_datum["conf"].numpy()
        pose_body = NumPyPoseBody(fps, pose_data, conf)

        return Pose(pose_header, pose_body)

    def _process_sentence(self, data_entry: Dict[str, Any]) -> ProcessedSentence:
        """
        Processes a single sentence entry from the DGS Corpus.
        Args:
            data_entry (Dict[str, Any]): A single sentence entry from the DGS Corpus.
        """
        original_poses_sequence = []
        updated_poses_sequence = []

        # Sentence data
        sentence = data_entry["sentence"]
        sentence_start_time = sentence["start"].numpy()
        # Gloss data
        glosses = sentence["glosses"]["gloss"].numpy()
        gloss_start_times = sentence["glosses"]["start"].numpy()
        gloss_end_times = sentence["glosses"]["end"].numpy()
        # Pose data
        pose_datum = data_entry["pose"]
        pose = self._create_pose_object(pose_datum, "dgs_corpus")
        pose_body = pose.body
        fps = pose_body.fps
        sentence_start_frame = math.floor(sentence_start_time / 1000 * fps)

        # Construct metadata dictionary
        metadata = {
            "document_id": data_entry["document_id"].numpy().decode("utf-8"),
            "id": data_entry["id"].numpy().decode("utf-8"),
            "sentence": {
                "id": data_entry["sentence"]["id"].numpy().decode("utf-8"),
                "start": data_entry["sentence"]["start"].numpy(),
                "end": data_entry["sentence"]["end"].numpy(),
                "english": data_entry["sentence"]["english"].numpy().decode("utf-8"),
                "german": data_entry["sentence"]["german"].numpy().decode("utf-8"),
                "glosses": {
                    "gloss": [
                        g.decode("utf-8")
                        for g in data_entry["sentence"]["glosses"]["gloss"].numpy()
                    ],
                    "start": data_entry["sentence"]["glosses"]["start"]
                    .numpy()
                    .tolist(),
                    "end": data_entry["sentence"]["glosses"]["end"].numpy().tolist(),
                },
            },
        }

        valid_gloss_count = 0
        replaced_gloss_count = 0
        for gloss, start, end in zip(glosses, gloss_start_times, gloss_end_times):
            gloss = gloss.decode("utf-8")
            start_frame = math.floor(start / 1000 * fps)
            end_frame = math.ceil(end / 1000 * fps)
            # Adjust gloss frame range relative to the sentence's start frame
            relative_start_frame = start_frame - sentence_start_frame
            relative_end_frame = end_frame - sentence_start_frame
            # print(f"Processing gloss: {gloss}")

            # Ensure relative frame range is within the sentence's local frame range
            if relative_start_frame < 0 or relative_end_frame > len(pose_body.data):
                print(f"Skipping Gloss '{gloss}' due to frame range out of bounds.")
                continue
            valid_gloss_count += 1

            # Extract original pose for current gloss
            original_pose_body = pose_body.select_frames(
                range(relative_start_frame, relative_end_frame)
            )
            original_pose_header = self._get_pose_header("dgs_corpus")
            original_pose = Pose(original_pose_header, original_pose_body)
            original_poses_sequence.append(original_pose)

            # Determine if the gloss should be replaced
            if self._should_replace(gloss):
                replaced_gloss_count += 1
                types_gloss_pose_path = self.dictionary[gloss]["views"]["pose"]
                with open(types_gloss_pose_path, "rb") as f:
                    types_gloss_pose = Pose.read(f.read())
                    updated_pose = types_gloss_pose
            else:
                updated_pose = original_pose
            updated_poses_sequence.append(updated_pose)

        # Ensure non-empty pose sequences
        if not original_poses_sequence or not updated_poses_sequence:
            logging.warning(
                "Skipping sentence %s due to empty pose sequences.", metadata["id"]
            )
            return None

        # Concatenate poses
        concatenated_original_pose = concatenate_poses(original_poses_sequence)
        concatenated_updated_pose = concatenate_poses(updated_poses_sequence)

        assert (
            concatenated_updated_pose.body.fps == concatenated_original_pose.body.fps
        ), (
            f"FPS mismatch: Original FPS = {concatenated_original_pose.body.fps}, "
            f"Updated FPS = {concatenated_updated_pose.body.fps}. "
            "Ensure FPS is consistent during loading."
        )

        return ProcessedSentence(
            concatenated_original_pose=concatenated_original_pose,
            concatenated_updated_pose=concatenated_updated_pose,
            metadata=metadata,
            valid_gloss_count=valid_gloss_count,
            replaced_gloss_count=replaced_gloss_count,
        )

    def generate_dataset(self, split: str, global_counter: Dict[str, int]):
        """
        Processes the specified split of the DGS Corpus and updates the pose sequences.
        Args:
            split (str): The data split to process ('train', 'validation', 'test').
            global_counter (Dict[str, int]): A dictionary to track the total processed examples globally.
        Yields:
            ProcessedSentence: The processed sentence data for each entry.
        """
        dgs_corpus = self._load_dgs_corpus()
        total_signs = 0
        replaced_signs = 0
        try:
            for _, data_entry in enumerate(dgs_corpus[split]):
                # Limit the number of examples to process
                if (
                    self.max_examples
                    and global_counter["processed"] >= self.max_examples
                ):
                    break
                # Process each sentence entry
                processed_sentence = self._process_sentence(data_entry)
                if processed_sentence is None:
                    continue
                # Update counters
                total_signs += processed_sentence.valid_gloss_count
                replaced_signs += processed_sentence.replaced_gloss_count
                global_counter["processed"] += 1
                print(
                    f"Processed {global_counter['processed']}/{self.max_examples or 'all'} sentences in total. "
                    f"Current split: {split}. Sentence ID: {processed_sentence.metadata['id']}"
                )
                yield processed_sentence
        finally:
            print(f"Processing complete for split: {split}.")
            print(f"Total signs: {total_signs}")
            print(f"Replaced signs: {replaced_signs}")


def save_pose_to_file(pose_object: Pose, file_path: Path):
    """
    Saves a Pose object to a file.
    Args:
        pose_object (Pose): The Pose object to save.
        file_path (Path): The file path to save the pose.
    """
    with open(file_path, "wb") as file:
        pose_object.write(file)


def convert_numpy_types(obj):
    """
    Recursively converts numpy types to standard Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


def save_metadata_to_file(metadata: Dict[str, Any], file_path: Path):
    """
    Saves metadata to a JSON file.
    Args:
        metadata (Dict[str, Any]): The metadata dictionary to save.
        file_path (Path): The file path to save the metadata.
    """
    metadata = convert_numpy_types(metadata)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def setup_logging(output_dir: Path, log_filename: str = "processing.log") -> Path:
    """Setup logging configuration."""
    log_file_path = Path(output_dir) / log_filename
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

    return log_file_path


def main():
    parser = argparse.ArgumentParser(description="Process and save pose dataset.")
    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Path to the DGS Corpus directory. If the dataset is not downloaded, "
        "it will be downloaded to this directory.",
    )
    parser.add_argument(
        "--dictionary_dir",
        type=str,
        required=True,
        help="Path to the DGS Types dictionary directory. If the dataset is not downloaded, "
        "it will be downloaded to this directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save processed pose files.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to process.",
    )
    args = parser.parse_args()

    # Initialize the DGSDataset class
    dgs_pose_dataset = DGSPoseDataset(
        corpus_dir=Path(args.corpus_dir),
        dictionary_dir=Path(args.dictionary_dir),
        max_examples=args.max_examples,
    )
    # Ensure output directory exists
    output_path = Path(args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    # Setup logging
    log_file_path = setup_logging(output_path)
    logging.info("Logging to file: %s", log_file_path)
    # Global counter to track total processed examples
    global_counter = {"processed": 0}

    # Iterate through the dataset splits
    for split in ["train", "validation", "test"]:
        split_output_path = output_path / split
        os.makedirs(split_output_path, exist_ok=True)
        logging.info("Starting processing for split: %s", split)
        # Process and save the dataset
        for idx, processed_sentence in enumerate(
            dgs_pose_dataset.generate_dataset(
                split=split, global_counter=global_counter
            )
        ):
            # Save original pose
            original_pose_path = split_output_path / f"{split}_{idx + 1}_original.pose"
            save_pose_to_file(
                processed_sentence.concatenated_original_pose, original_pose_path
            )
            # Save updated pose
            updated_pose_path = split_output_path / f"{split}_{idx + 1}_updated.pose"
            save_pose_to_file(
                processed_sentence.concatenated_updated_pose, updated_pose_path
            )
            # Save metadata
            metadata_path = split_output_path / f"{split}_{idx + 1}_metadata.json"
            save_metadata_to_file(processed_sentence.metadata, metadata_path)

            # Log processing information
            logging.info(
                "Processed sentence %d: %s", idx + 1, processed_sentence.metadata["id"]
            )
            logging.info("Original pose saved to: %s", original_pose_path)
            logging.info("Updated pose saved to: %s", updated_pose_path)
            logging.info("Metadata saved to: %s", metadata_path)

            valid_gloss_count = processed_sentence.valid_gloss_count
            replaced_gloss_count = processed_sentence.replaced_gloss_count

            logging.info("Valid gloss count: %d", valid_gloss_count)
            logging.info("Replaced gloss count: %d", replaced_gloss_count)
            logging.info("Metadata: %s", processed_sentence.metadata)

            # Break if maximum examples is reached
            if args.max_examples and global_counter["processed"] >= args.max_examples:
                break


if __name__ == "__main__":
    main()
