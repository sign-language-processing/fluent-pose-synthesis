from typing import Dict, List, Tuple, Any
import tensorflow as tf


# Define types
PoseDict = Dict[str, Dict[str, Any]]
ConflictDict = Dict[str, List[Dict[str, Any]]]

def create_gloss_to_pose_dict(dgs_types_dataset: tf.data.Dataset) -> Tuple[PoseDict, ConflictDict]:
    """
    Generate a mapping of glosses to their corresponding pose data from DGS Types dataset.
    """
    gloss_to_pose_dict = {}
    gloss_pose_conflicts = {}
    
    for datum in dgs_types_dataset["train"]:
        # Get glosses and id
        glosses = datum['glosses'].numpy().tolist()
        datum_id = datum['id'].numpy().decode('utf-8')
        is_galex = datum_id.startswith('galex_')
        # Get pose and view names
        view_names = datum['views']['name'].numpy().tolist()
        poses = datum['views']['pose'].numpy().tolist()
        if len(poses) == 0:
            pose = None
        elif len(poses) == 1:
            pose = poses[0] # Use the only pose available (which is front)
        else:
            frontal_index = view_names.index(b'frontal')  # Get the frontal view index
            pose = poses[frontal_index] # Use the frontal view pose

        # Keep complete original data
        original_data = {k: v for k, v in datum.items()}
        
        # Add pose for each gloss
        for gloss in glosses:
            # Convert gloss to string if it's bytes
            gloss = gloss.decode('utf-8') if isinstance(gloss, bytes) else gloss
            
            # Check if a pose already exists for this gloss
            if gloss in gloss_to_pose_dict:
                existing_pose = gloss_to_pose_dict[gloss]['views']['pose']
                existing_id = gloss_to_pose_dict[gloss]['id'].numpy().decode('utf-8')
                existing_is_galex = existing_id.startswith('galex_')
                
                # Case 1: If existing pose is None, update with new pose
                if existing_pose is None and pose is not None:
                    gloss_to_pose_dict[gloss]['views']['pose'] = pose                   
                # Case 2: If new pose is None, keep existing pose
                elif pose is None:
                    pass
                # Case 3: If both poses exist and are different
                elif pose != existing_pose:
                    # Record conflict for debugging
                    if gloss not in gloss_pose_conflicts:
                        gloss_pose_conflicts[gloss] = []
                    gloss_pose_conflicts[gloss].append({
                        'pose': pose,
                        'id': datum_id,
                        'type': 'Galex' if is_galex else 'DGS Types'
                    })                                      
                    # Replace if existing entry is from Galex and new one is from DGS Types
                    # Otherwise (both are from the same source or new one is from Galex), keep the existing entry
                    if existing_is_galex and not is_galex:
                        gloss_to_pose_dict[gloss] = {
                            **original_data,
                            'views': {**original_data['views'], 'pose': pose}
                        }
            # Create new entry if gloss doesn't exist
            else:
                gloss_to_pose_dict[gloss] = {
                    **original_data,
                    'views': {**original_data['views'], 'pose': pose}
                }

    # # Print conflicts summary
    # print("\n--- Gloss-Pose Conflicts Summary ---")
    # for gloss, conflicts in gloss_pose_conflicts.items():
    #     print(f"\nGloss '{gloss}' had conflicts:")
    #     first_entry = gloss_to_pose_dict[gloss]
    #     first_id = first_entry['id'].numpy().decode('utf-8')
    #     print(f"  Final chosen pose (ID: {first_id}):")
    #     print(f"    {first_entry['views']['pose']}")
    #     print("  Conflicting entries:")
    #     for conflict in conflicts:
    #         print(f"    - {conflict['type']} (ID: {conflict['id']})")
    #         print(f"      Pose: {conflict['pose']}")
    
    return gloss_to_pose_dict, gloss_pose_conflicts