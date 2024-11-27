import os
import itertools
import importlib

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import lru_cache
import math

from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus import DgsCorpusConfig
import sign_language_datasets.datasets

from pose_format import PoseHeader, Pose
from pose_format.numpy.pose_body import NumPyPoseBody, PoseBody
from pose_format.tensorflow.pose_body import TensorflowPoseBody
from pose_format.utils.reader import BufferReader
from spoken_to_signed.gloss_to_pose import concatenate_poses

# Project-Specific Imports
from dgs_types_mapping import create_gloss_to_pose_dict




class DGSPoseDataset:
    def __init__(self, corpus_dir, dictionary_dir, num_examples=None):
        """
        Initializes the DGSPoseDataset class.
        
        :param corpus_dir: Directory to load DGS Corpus.
        :param dictionary_dir: Directory to load DGS Types.
        :param num_examples: Maximum number of examples to process (default: None, process all examples).
        """
        self.num_examples = num_examples
        self.corpus_dir = corpus_dir
        self.dictionary_dir = dictionary_dir
        self.corpus = self._load_dgs_corpus()
        self.dictionary = self._load_dgs_types_dictionary()
        self.dataset = []
        self.total_signs = 0
        self.replaced_signs = 0
        # Cache for pose headers to avoid repeated loading
        self._pose_headers_cache = {}

    def _load_dgs_corpus(self):
        """
        Loads the DGS Corpus dataset.
        """
        config_corpus = DgsCorpusConfig(
            name="sentence-level-pose",
            version="3.0.0",
            include_video=False,
            include_pose="holistic",
            data_type="sentence"
        )
        return tfds.load('dgs_corpus',
                         builder_kwargs={"config": config_corpus},
                         data_dir=self.corpus_dir)

    def _load_dgs_types_dictionary(self):
        """
        Loads the DGS Types dataset and creates a gloss-to-pose dictionary.
        """
        config = SignDatasetConfig(
            name="pose_holistic",
            version="3.0.0",
            include_video=False,
            include_pose="holistic",
            process_pose=False
        )
        dgs_types = tfds.load('dgs_types', 
                              builder_kwargs=dict(config=config), 
                              data_dir=self.dictionary_dir)
        dgs_types_dict, _ = create_gloss_to_pose_dict(dgs_types)
        return dgs_types_dict

    def _should_replace(self, gloss):
        """
        Determines whether a gloss's pose should be replaced.
        """
        # Check if gloss is in dgs types dictionary and has a pose, and is not a special generic sign
        return not gloss.startswith('$') and gloss in self.dictionary and self.dictionary[gloss] is not None

    @lru_cache(maxsize=None)
    def _get_pose_header(self, dataset_name: str):
        """
        Retrieves the pose header for a given dataset.
        :param dataset_name: Name of the dataset (e.g., "dgs_corpus").
        :return: PoseHeader object.
        """
        try:
            # Check cache first
            if dataset_name in self._pose_headers_cache:
                return self._pose_headers_cache[dataset_name]

            # Dynamically import the dataset module
            dataset_module = importlib.import_module(f"sign_language_datasets.datasets.{dataset_name}.{dataset_name}")
            
            # Read the pose header from the dataset's predefined file
            with open(dataset_module._POSE_HEADERS["holistic"], "rb") as buffer:
                pose_header = PoseHeader.read(BufferReader(buffer.read()))
            
            # Cache the pose header
            self._pose_headers_cache[dataset_name] = pose_header
            return pose_header
        except Exception as e:
            print(f"Error getting pose header for {dataset_name}: {e}")
            raise

    def _create_pose_object(self, pose_datum, dataset_name: str):
        """
        Creates a Pose object from the given pose data and dataset name.
        :param pose_datum: Pose data dictionary containing fps, data, and conf.
        :param dataset_name: Name of the dataset.
        :return: Pose object.
        """
        try:
            # Retrieve the pose header for the dataset
            pose_header = self._get_pose_header(dataset_name)
            
            # Extract pose data
            fps = float(pose_datum["fps"].numpy())
            pose_data = pose_datum["data"].numpy()
            conf = pose_datum["conf"].numpy()
            pose_body = NumPyPoseBody(fps, pose_data, conf)
            
            # Construct and return the Pose object
            pose = Pose(pose_header, pose_body)
            return pose
        except Exception as e:
            print(f"Error creating pose object for {dataset_name}: {e}")
            raise

    def _process_sentence(self, sentence_entry):
        """
        Processes a single sentence entry from the DGS Corpus.

        :param sentence_entry: A single sentence entry from the DGS Corpus.
        :return: Dictionary with sentence ID, updated pose, original pose, and metadata.
        """
        sentence = sentence_entry['sentence']
        glosses = sentence['glosses']['gloss'].numpy()
        gloss_start_times = sentence['glosses']['start'].numpy()
        gloss_end_times = sentence['glosses']['end'].numpy()

        pose_datum = sentence_entry['pose']
        pose_object = self._create_pose_object(pose_datum, "dgs_corpus")
        pose_data = pose_datum['data'].numpy() # (Frames, People, Points, Dims)
        fps = pose_datum['fps'].numpy()
        conf = pose_datum['conf'].numpy()
        pose_body = NumPyPoseBody(fps, pose_data, conf)

        updated_gloss_poses = []
        original_gloss_poses = []


        # metadata = {
        #     "sentence_id": sentence['id'].numpy().decode('utf-8'),
        #     "glosses": [],
        #     "gloss_start_times": gloss_start_times.tolist(),
        #     "gloss_end_times": gloss_end_times.tolist()
        # }

        for gloss, start, end in zip(glosses, gloss_start_times, gloss_end_times):
            gloss = gloss.decode('utf-8')
            start_frame = math.floor(start / 1000 * fps)
            end_frame = math.ceil(end / 1000 * fps)
            print(f"fps: {fps}")
            print(f"start_frame: {start_frame}, end_frame: {end_frame}, data_size: {len(pose_body.data)}")
            ### Error: start_frame: 299, end_frame: 304, data_size: 133

            # Extract original pose for current gloss
            original_pose_body = pose_body.select_frames(range(start_frame, end_frame))
            original_pose_header = self._get_pose_header("dgs_corpus")
            original_pose = Pose(original_pose_header, original_pose_body)
            original_gloss_poses.append(original_pose)


            # Determine if the gloss should be replaced
            if self._should_replace(gloss):
                types_gloss_pose_path = self.dictionary[gloss]['views']['pose']
                with open(types_gloss_pose_path, "rb") as f:
                    types_gloss_pose = Pose.read(f.read())
                    gloss_pose = types_gloss_pose
                self.replaced_signs += 1
            else:
                gloss_pose = original_pose


            updated_gloss_poses.append(gloss_pose)
            # metadata["glosses"].append(gloss)

        # Concatenate poses
        updated_pose = concatenate_poses(updated_gloss_poses)
        original_pose = concatenate_poses(original_gloss_poses)

        return {
            "sentence_id": metadata["sentence_id"],
            "updated_pose": updated_pose,
            "original_pose": original_pose,
            "metadata": metadata
        }

    # def generate_dataset(self):
    #     """
    #     Processes the entire DGS Corpus and updates the pose sequences.
    #     """
    #     print("Processing DGS Corpus...")
    #     for i, sentence_entry in enumerate(self.corpus['train']):
    #         # Limit number of examples to process
    #         if self.num_examples and i >= self.num_examples:
    #             break
    #         processed_data = self._process_sentence(sentence_entry)
    #         self.dataset.append(processed_data)
    #         self.total_signs += len(processed_data["updated_pose"])
    #     print(f"Processed {len(self.dataset)} sentences.")
    #     print(f"Total signs: {self.total_signs}")
    #     print(f"Replaced signs: {self.replaced_signs}")
    #     return self.dataset

    # def get_metadata(self):
    #     """
    #     Retrieves metadata for the entire dataset.
    #     """
    #     return {
    #         "corpus_metadata": self.corpus["metadata"],
    #         "dictionary_metadata": self.dictionary,
    #         "processed_metadata": [data["metadata"] for data in self.dataset]
    #     }