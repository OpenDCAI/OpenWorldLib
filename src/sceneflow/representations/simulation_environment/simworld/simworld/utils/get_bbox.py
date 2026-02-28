"""Get the bounding box of the selected actor and save it to a JSON file."""
import json

import unreal

bbox_dict = {}
file_path = '<path to your bounding_boxes.json>'

selected_actors = unreal.EditorLevelLibrary.get_selected_level_actors()

if selected_actors:
    for selected_actor in selected_actors:
        if isinstance(selected_actor, unreal.Actor):
            origin, box_extent = selected_actor.get_actor_bounds(False)

            min_point = unreal.Vector(origin.x - box_extent.x, origin.y - box_extent.y, origin.z - box_extent.z)
            max_point = unreal.Vector(origin.x + box_extent.x, origin.y + box_extent.y, origin.z + box_extent.z)
            bbox = (max_point - min_point)
            print(f"bbox: {{'x': {bbox.x}, 'y': {bbox.y}, 'z': {bbox.z}}}")
            label = selected_actor.get_class().get_name()
            bbox_dict[f'{label}'] = {
                'bbox': {
                    'x': bbox.x,
                    'y': bbox.y,
                    'z': bbox.z
                }
            }
        else:
            print('skip non-Actor object')
else:
    print('No actor selected.')

# write new JSON data directly
with open(file_path, 'w') as f:
    json.dump(bbox_dict, f, indent=4, ensure_ascii=False)
