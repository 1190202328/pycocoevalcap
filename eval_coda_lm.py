import argparse

from pycocotools.coco import COCO

from eval import COCOEvalCap


def get_score(annotation_file, results_file, selected_metrics=None):
    # create coco object and coco_result object
    if selected_metrics is None:
        selected_metrics = {'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L'}
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    results_dict = {}
    num = 0
    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        if metric in selected_metrics:
            score = round(score * 100, 2)
            results_dict[metric] = score
            num += score
    num /= len(results_dict)
    results_dict['avg'] = round(num, 2)

    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert2coco")
    parser.add_argument("--directory", type=str)
    args = parser.parse_args()

    gt_directory = '/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/CODA-LM/NEW_Mini/vqa_anno_ans/coco_format'
    pred_directory = f'{args.directory}/coco_format'

    json_names = ['general_perception', 'driving_suggestion']

    total_dict = {}
    for json_name in json_names:
        print('-' * 10 + f'evaluating [{json_name}]' + '-' * 10)
        annotation_file = f'{gt_directory}/{json_name}_answer_coco.json'
        results_file = f'{pred_directory}/{json_name}_answer_coco.json'
        results_dict = get_score(annotation_file, results_file)
        print(results_dict)
        total_dict[json_name] = results_dict['avg']

    json_names_region = ['vehicle', 'vru',
                         'traffic_sign', 'traffic_light',
                         'traffic_cone',
                         'barrier', 'miscellaneous']
    total_dict['region_perception'] = {}
    for json_name in json_names_region:
        print('-' * 10 + f'evaluating [{json_name}]' + '-' * 10)
        annotation_file = f'{gt_directory}/region_perception_{json_name}_answer_coco.json'
        results_file = f'{pred_directory}/region_perception_{json_name}_answer_coco.json'
        results_dict = get_score(annotation_file, results_file)
        print(results_dict)
        total_dict['region_perception'][json_name] = results_dict['avg']

    for key in total_dict:
        if key in json_names:
            print(f"{key}: {total_dict[key]}")

    print('region_perception:')
    for key in total_dict['region_perception']:
        if key in json_names_region:
            print(f"\t{key}: {total_dict['region_perception'][key]}")
