import json

def create_blessing_data(conversations_list):
    """
    将对话数据处理成指定的格式
    
    参数:
    conversations_list: 包含多个对话的列表，每个对话包含system、input、output
    
    返回:
    格式化后的数据列表
    """
    result = []
    
    for conversation_data in conversations_list:
        conversation_data = conversation_data.get("conversation", {})[0]
        formatted_conversation = {
                "instruction": conversation_data.get("system", ""),
                "input": conversation_data.get("input", ""),
                "output": conversation_data.get("output", "")
        }
        result.append(formatted_conversation)
    
    return result



def save_to_json(data, filename="blessing_data.json"):
    """保存数据到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"数据已保存到 {filename}")


if __name__ == "__main__":
    file_path = './dataset/tianji-chinese/tianji-wishes-chinese-v0.1.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_conversations = json.load(f)
    
    # get formatted data
    formatted_data = create_blessing_data(raw_conversations)
    
    # save formatted data to JSON file
    save_to_json(formatted_data, filename="./dataset/tianji-chinese/tianji-wishes-chinese-v0.1-format.json")