#!/bin/bash

prediction_dir=$1

root=/nfs/ofs-902-1/object-detection/jiangjing/experiments/CODA-LM/evaluation
coco_root=/nfs/ofs-902-1/object-detection/jiangjing/experiments/pycocoevalcap
reference_dir=/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/CODA-LM/NEW_Mini

echo "---step1: convert2eval---"
cd "${root}" && python \
  convert2eval.py --reference_path "${reference_dir}" --prediction_path "${prediction_dir}/region_perception_answer.jsonl"

echo "---step2: convert into coco format---"
cd "${coco_root}" && python convert2coco_ans.py --directory "${prediction_dir}"

echo "---step3: eval---"
python eval_coda_lm.py --directory "${prediction_dir}"
