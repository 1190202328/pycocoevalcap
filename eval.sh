#!/bin/bash

#bash /nfs/volume-902-16/tangwenbo/s3_all.sh

prediction_dir=$1

root=/Users/didi/Desktop/CODA-LM/evaluation
coco_root=/Users/didi/Desktop/pycocoevalcap
reference_dir=/Users/didi/Desktop/ECCV比赛/验证数据保存/NEW_Mini
save_dir=/Users/didi/Downloads/Z_WorkShop

echo "---step1: convert2eval---"
cd "${root}" && python \
  convert2eval.py --reference_path "${reference_dir}" --prediction_path "${prediction_dir}/region_perception_answer.jsonl"

echo "---step2: convert into coco format---"
cd "${coco_root}" && python convert2coco.py --directory /Users/didi/Desktop/ECCV比赛/验证数据保存/NEW_Mini/vqa_anno_ans
cd "${coco_root}" && python convert2coco_ans.py --directory "${prediction_dir}"

echo "---step3: eval---"
python eval_coda_lm.py --directory "${prediction_dir}"
