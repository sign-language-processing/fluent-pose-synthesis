import os
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
from pose_format.numpy.pose_body import NumPyPoseBody
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
        return not gloss.startswith('$') and gloss in self.dictionary and self.dictionary[gloss]['views']['pose'] is not None

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
        Creates a Pose object from the given pose data_entry and dataset name.
        :param pose_datum: Pose data_entry dictionary containing fps, data_entry, and conf.
        :param dataset_name: Name of the dataset.
        :return: Pose object.
        """
        try:
            # Retrieve the pose header for the dataset
            pose_header = self._get_pose_header(dataset_name)
            
            # Extract pose data_entry
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

    def _process_sentence(self, data_entry):
        """
        Processes a single sentence entry from the DGS Corpus.

        :param data_entry: A single sentence entry from the DGS Corpus.
        :return: Dictionary with sentence ID, updated pose, original pose, and metadata.
        """
        
        # Sentence data
        sentence = data_entry['sentence']
        sentence_id = sentence['id'].numpy()
        sentence_start_time = sentence['start'].numpy()
        sentence_end_time = sentence['end'].numpy()

        glosses = sentence['glosses']['gloss'].numpy()
        gloss_start_times = sentence['glosses']['start'].numpy()
        gloss_end_times = sentence['glosses']['end'].numpy()

        # Pose data
        pose_datum = data_entry['pose']
        pose_data = pose_datum['data'].numpy() # (Frames, People, Points, Dims)
        fps = pose_datum['fps'].numpy()
        conf = pose_datum['conf'].numpy()
        pose_body = NumPyPoseBody(fps, pose_data, conf)

        original_poses_sequence = []
        updated_poses_sequence = []

        # Calculate sentence frame range
        sentence_start_frame = math.floor(sentence_start_time / 1000 * fps)
        sentence_end_frame = math.ceil(sentence_end_time / 1000 * fps)
        sentence_frame_range = range(sentence_start_frame, sentence_end_frame)

        # Construct metadata dictionary
        metadata = {
            "document_id": data_entry['document_id'].numpy().decode('utf-8'),
            "id": data_entry['id'].numpy().decode('utf-8'),
            "paths": {
                "cmdi": data_entry['paths']['cmdi'].numpy().decode('utf-8'),
                "eaf": data_entry['paths']['eaf'].numpy().decode('utf-8'),
                "ilex": data_entry['paths']['ilex'].numpy().decode('utf-8'),
                "srt": data_entry['paths']['srt'].numpy().decode('utf-8'),
            },
            "pose": {
                "fps": data_entry['pose']['fps'].numpy(),
                "data": data_entry['pose']['data'].numpy(),
                "conf": data_entry['pose']['conf'].numpy()
            },
            "sentence": {
                "id": data_entry['sentence']['id'].numpy().decode('utf-8'),
                "start": data_entry['sentence']['start'].numpy(),
                "end": data_entry['sentence']['end'].numpy(),
                "english": data_entry['sentence']['english'].numpy().decode('utf-8'),
                "german": data_entry['sentence']['german'].numpy().decode('utf-8'),
                "glosses": {
                    "Gebärde": [g.decode('utf-8') for g in data_entry['sentence']['glosses']['Gebärde'].numpy()],
                    "Lexeme_Sign": [g.decode('utf-8') for g in data_entry['sentence']['glosses']['Lexeme_Sign'].numpy()],
                    "Sign": [g.decode('utf-8') for g in data_entry['sentence']['glosses']['Sign'].numpy()],
                    "gloss": [g.decode('utf-8') for g in data_entry['sentence']['glosses']['gloss'].numpy()],
                    "start": data_entry['sentence']['glosses']['start'].numpy().tolist(),
                    "end": data_entry['sentence']['glosses']['end'].numpy().tolist(),
                    "hand": [h.decode('utf-8') for h in data_entry['sentence']['glosses']['hand'].numpy()],
                },
                "mouthings": {
                    "mouthing": [m.decode('utf-8') for m in data_entry['sentence']['mouthings']['mouthing'].numpy()],
                    "start": data_entry['sentence']['mouthings']['start'].numpy().tolist(),
                    "end": data_entry['sentence']['mouthings']['end'].numpy().tolist(),
                },
                "participant": data_entry['sentence']['participant'].numpy().decode('utf-8'),
            }
        }

        valid_gloss_count = 0
        replaced_gloss_count = 0

        for gloss, start, end in zip(glosses, gloss_start_times, gloss_end_times):
            gloss = gloss.decode('utf-8')
            start_frame = math.floor(start / 1000 * fps)
            end_frame = math.ceil(end / 1000 * fps)

            # Adjust gloss frame range relative to the sentence's start frame
            relative_start_frame = start_frame - sentence_start_frame
            relative_end_frame = end_frame - sentence_start_frame

            print(f"Processing sentence id: {sentence_id}")
            print(f"Processing gloss: {gloss}")
            # print(f"absolute_start_frame: {start_frame}, absolute_end_frame: {end_frame}, data_size: {len(pose_body.data)}")
            print(f"relative_start_frame: {relative_start_frame}, relative_end_frame: {relative_end_frame}")

            # Ensure relative frame range is within the sentence's local frame range
            if relative_start_frame < 0 or relative_end_frame > len(pose_body.data):
                print(f"Skipping Gloss '{gloss}' due to frame range out of bounds.")
                continue

            valid_gloss_count += 1


            # Extract original pose for current gloss
            original_pose_body = pose_body.select_frames(range(relative_start_frame, relative_end_frame))
            original_pose_header = self._get_pose_header("dgs_corpus")
            original_pose = Pose(original_pose_header, original_pose_body)
            original_poses_sequence.append(original_pose)


            # Determine if the gloss should be replaced
            if self._should_replace(gloss):
                print(f"Original pose shape: {original_pose_body.data.shape}")
                replaced_gloss_count += 1
                types_gloss_pose_path = self.dictionary[gloss]['views']['pose']
                with open(types_gloss_pose_path, "rb") as f:
                    types_gloss_pose = Pose.read(f.read())

                    # original_frame_count = len(original_pose_body.data)
                    # current_frame_count = len(types_gloss_pose.body.data)
                    # print(f"Original frame count: {original_frame_count}")
                    # print(f"Current frame count: {current_frame_count}")
                    print(f"Updated pose shape before interpolation: {types_gloss_pose.body.data.shape}")

                    # Interpolate pose if fps of dgs types pose is different from dgs corpus pose
                    if types_gloss_pose.body.fps != original_pose_body.fps:
                        interpolated_pose_body = types_gloss_pose.body.interpolate(new_fps=original_pose_body.fps)
                        updated_pose_header = self._get_pose_header("dgs_types")
                        updated_pose = Pose(updated_pose_header, interpolated_pose_body)
                        print(f"Updated pose shape after interpolation: {interpolated_pose_body.data.shape}")
                    else:
                        updated_pose = types_gloss_pose
            else:
                updated_pose = original_pose

            updated_poses_sequence.append(updated_pose)


        # Concatenate poses
        concatenated_original_pose = concatenate_poses(original_poses_sequence)
        concatenated_updated_pose = concatenate_poses(updated_poses_sequence)

        # print(f"Concatenated original pose shape: {concatenated_original_pose.body.data.shape}")
        # print(f"Concatenated updated pose shape: {concatenated_updated_pose.body.data.shape}")

        return concatenated_original_pose, concatenated_updated_pose, metadata, valid_gloss_count, replaced_gloss_count

    def generate_dataset(self):
        """
        Processes the entire DGS Corpus and updates the pose sequences.
        Returns the processed dataset.
        """
        print("Processing DGS Corpus...")

        # Initialize counters and dataset list
        self.dataset = []
        self.total_signs = 0
        self.replaced_signs = 0

        for idx, data_entry in enumerate(self.corpus['train']):
            # Limit number of examples to process
            if self.num_examples and idx >= self.num_examples:
                break

            # Process each sentence entry
            processed_data = self._process_sentence(data_entry)
            self.dataset.append(processed_data)

            # Update counters
            self.total_signs += processed_data[3]
            self.replaced_signs += processed_data[4]

            # Log progress
            print(f"Processed {idx + 1}/{self.num_examples or 'all'} sentences.")
            print(f"Total signs: {self.total_signs}")
            print(f"Replaced signs: {self.replaced_signs}")


        # Final summary
        print("Processing complete.")
        print(f"Total sentences processed: {len(self.dataset)}")
        print(f"Total signs: {self.total_signs}")
        print(f"Replaced signs: {self.replaced_signs}")

        return self.dataset




# # Initialize the DGSDataset class
# dgs_pose_dataset = DGSPoseDataset(
#     corpus_dir='/scratch/ronli',
#     dictionary_dir='/scratch/ronli',
#     num_examples=3
# )

# # Generate the dataset
# processed_data_list = dgs_pose_dataset.generate_dataset()


# def save_pose_to_file(pose_object, file_path):
#     with open(file_path, "wb") as file:
#         pose_object.write(file)

# os.makedirs('output', exist_ok=True)

# # Iterate through the processed sentences and save poses
# for sentence_index, sentence_data in enumerate(processed_data_list):
#     concatenated_original_pose, concatenated_updated_pose, metadata, _, _ = sentence_data

#     print(f"Saving poses for sentence {sentence_index}...")
#     print(f"Concatenated original pose shape: {concatenated_original_pose.body.data.shape}")
#     print(f"Concatenated updated pose shape: {concatenated_updated_pose.body.data.shape}")


    
#     # Save the original and updated poses to files
#     save_pose_to_file(concatenated_original_pose, os.path.join('output', f"original_pose_{sentence_index}.pose"))
#     save_pose_to_file(concatenated_updated_pose, os.path.join('output', f"updated_pose_{sentence_index}.pose"))
    
#     # Print the document id, sentence id and glosses
#     print(f"Document ID: {metadata['document_id']}")
#     print(f"Sentence ID: {metadata['sentence']['id']}")
#     print(f"English: {metadata['sentence']['english']}")
#     print(f"German: {metadata['sentence']['german']}")
#     print(f"Glosses: {metadata['sentence']['glosses']['gloss']}")
#     print(f"Metadata: {metadata}")


