import argparse
import json
import os

label_dict = {
    "vehicle": {"car", "truck", "tram", "tricycle", "bus", "trailer", "construction_vehicle",
                "recreational_vehicle"},
    "vru": {"pedestrian", "cyclist", "bicycle", "moped", "motorcycle", "stroller", "wheelchair", "cart"},
    "traffic_sign": {"warning_sign", "traffic_sign"},
    "traffic_light": {"traffic_light"},
    "traffic_cone": {"traffic_cone"},
    "barrier": {"barrier", "bollard"},
    "miscellaneous": {"dog", "cat", "sentry_box", "traffic_box", "traffic_island", "debris", "suitcace",
                      "dustbin", "concrete_block", "machinery", "chair", "phone_booth", "basket", "cardboard",
                      "carton", "garbage", "garbage_bag", "plastic_bag", "stone", "tire", "misc"},
}


def get_big_label(label):
    for big_label in label_dict:
        if label in label_dict[big_label]:
            return big_label
    raise Exception


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert2coco")
    parser.add_argument("--directory", type=str)
    args = parser.parse_args()

    directory = args.directory
    save_directory = f'{directory}/coco_format'
    os.makedirs(save_directory, exist_ok=True)

    anno_list = ['general_perception_answer', 'driving_suggestion_answer']

    for anno_name in anno_list:
        coco_json = {
            'info': anno_name,
            'images': [],
            'annotations': [],
            'licenses': []
        }
        jsonl_path = f'{directory}/{anno_name}.jsonl'
        with open(jsonl_path, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                line_json = json.loads(line.strip())
                coco_json['images'].append({
                    'id': line_json['image'],
                    'file_name': line_json['image']
                })
                coco_json['annotations'].append({
                    "image_id": line_json['image'],
                    "id": line_json['image'],
                    "caption": line_json['answer']
                })
        with open(f'{save_directory}/{anno_name}_coco.json', mode='w', encoding='utf-8') as f:
            f.write(json.dumps(coco_json))

    jsonl_path = f'{directory}/region_perception_answer_w_label.jsonl'
    class_anno_dict = {}
    with open(jsonl_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            line_json = json.loads(line.strip())
            big_label = get_big_label(line_json['label_name'])
            if big_label not in class_anno_dict:
                class_anno_dict[big_label] = {
                    'images': [],
                    'annotations': [],
                }
            class_anno_dict[big_label]['images'].append({
                'id': line_json['image'],
                'file_name': line_json['image']
            })
            class_anno_dict[big_label]['annotations'].append({
                "image_id": line_json['image'],
                "id": line_json['image'],
                "caption": line_json['answer']
            })

    for big_label in class_anno_dict:
        with open(f'{save_directory}/region_perception_{big_label}_answer_coco.json', mode='w',
                  encoding='utf-8') as f:
            f.write(json.dumps(class_anno_dict[big_label]))
