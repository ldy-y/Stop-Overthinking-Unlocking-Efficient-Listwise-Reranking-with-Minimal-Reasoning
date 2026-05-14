import json

file_path = "/mnt/ali-sh-1/usr/ningyikai/ldy_file/Rank/TestScaling/train_data/ms-marco-data/distil-processed/Rank-K-32B_msmarco_depth_20_N_16_T_0.70_RP_1.10_maxlen_8192_random_17643142119094843.jsonl"

with open(file_path, 'r', encoding='utf-8') as f:
    first_line = f.readline().strip()
    if first_line:
        data = json.loads(first_line)
        print(type(data))
        print(data.keys())
        # print(data['doc_ids'])
        # print(data['scores'][0])
        # print(data['model_responses'][1])
        # print(data['cleared_model_responses'][0])
        # print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("文件为空或第一行为空")
