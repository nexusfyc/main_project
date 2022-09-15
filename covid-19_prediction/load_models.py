import torch
from transformers import BertTokenizer, BertModel

# 根据模型文件目录加载
bert_path = '/Library/RoBERTa_zh_Large_PyTorch'
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert = BertModel.from_pretrained(bert_path, return_dict=True)



# 若加载bert模型时return_dict=False，则输出last_hidden_state与pooler_output组成的元组
text_list = ["上海新增1例本土确诊病例轨迹",
             "据上海市新冠肺炎疫情防控新闻发布会，上海市新增1例新冠肺炎本土确诊病例。该病例涉及的轨迹为上海市普陀区甘泉街道志丹路155号西部名都花园，石泉街道宁强路33号石泉社区文化活动中心"]
inputs = tokenizer(text_list, return_tensors="pt", padding=True)  # 长度不等时会进行填充
outputs = bert(**inputs)

print(outputs[0].shape)


