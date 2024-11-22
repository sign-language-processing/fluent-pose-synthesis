import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig


config = SignDatasetConfig(
   name="pose_holistic",
   version="3.0.0",
   include_video=False,
   include_pose="holistic",
   process_pose=False
)

# Load dataset
dgs_types = tfds.load('dgs_types',
   builder_kwargs=dict(config=config),
   data_dir='/scratch/ronli')


def create_gloss_to_pose_dict(dgs_types_dataset):
   gloss_to_pose_dict = {}
   gloss_pose_conflicts = {}
   
   for datum in dgs_types_dataset["train"]:
       # Get glosses
       glosses = datum['glosses'].numpy().tolist()
       
       # Get pose (if exists)
       pose = datum['views']['pose'].numpy().tolist()
       pose = pose[0] if pose else None
       
    #    # Print debug info
    #    print(f"Processing datum:")
    #    print(f"  Glosses: {glosses}")
    #    print(f"  Pose: {pose}")
       
       # Keep complete original data
       original_data = {k: v for k, v in datum.items()}
       
       # Add pose for each gloss
       for gloss in glosses:
           # Convert gloss to string if it's bytes
           gloss = gloss.decode('utf-8') if isinstance(gloss, bytes) else gloss
           
           # Check if a pose already exists for this gloss
           if gloss in gloss_to_pose_dict:
               existing_pose = gloss_to_pose_dict[gloss]['views']['pose']
               
               # If existing pose is None, update with new pose
               if existing_pose is None and pose is not None:
                   gloss_to_pose_dict[gloss]['views']['pose'] = pose
                   print(f"  Updated None pose for gloss '{gloss}' with: {pose}")
                   continue                  
               # If new pose is None, keep existing pose
               elif pose is None:
                   continue                  
               # If both poses exist and are different, record conflict
               elif pose != existing_pose:
                   if gloss not in gloss_pose_conflicts:
                       gloss_pose_conflicts[gloss] = [existing_pose]
                   
                   print(f"  CONFLICT for gloss '{gloss}':")
                   print(f"    Existing pose: {existing_pose}")
                   print(f"    New pose: {pose}")
                   gloss_pose_conflicts[gloss].append(pose)
                   
                   # Keep first non-None pose
                   continue
           
           gloss_to_pose_dict[gloss] = {
               **original_data,
               'views': {**original_data['views'], 'pose': pose}
           }
   
   # Print conflicts summary
   print("\n--- Gloss-Pose Conflicts Summary ---")
   for gloss, conflicting_poses in gloss_pose_conflicts.items():
       print(f"Gloss '{gloss}' has multiple poses:")
       for pose in conflicting_poses:
           print(f"  - {pose}")
   
   return gloss_to_pose_dict, gloss_pose_conflicts



# Create gloss to pose mapping
gloss_pose_dict, conflicts = create_gloss_to_pose_dict(dgs_types)

# Print example mappings
print("\n--- Example Gloss-Pose Mappings ---")
for i, (gloss, data) in enumerate(gloss_pose_dict.items()):
   print(f"Gloss {i+1}: {gloss}")
   print(f"  Pose: {data['views']['pose']}")
   print(f"  ID: {data['id'].numpy().decode('utf-8')}")
   print("---")
   
   # Limit print output
   if i >= 4:
       print("Only showing first 5 entries...")
       break

# Print conflict statistics
print(f"\nTotal unique glosses: {len(gloss_pose_dict)}")
print(f"Glosses with conflicting poses: {len(conflicts)} (excluding None conflicts)")