import fire
from llama import Llama
from typing import List, Tuple, Dict
import networkx as nx
import matplotlib.pyplot as plt

title="Hospital Workflow",
description = "A hospital wants to establish a rating workflow for their doctors. To make the workflow reliable two different roles are assigned. The first one is a referee from the newly created quality assurance department while the second one represents the managing director of the hospital. Both roles execute all of their tasks independently from each other. The referee starts a new case regarding a certain doctor by interviewing patients. Since a patient interview workflow is already established, it is simply integrated in the new workflow. Meanwhile, the director asks an external expert to review the work of the doctor under rating. Unfortunately, since the expert only gets a low expenses fee, it can happen that the expert is not responding in time. If that happens, another expert has to be asked (who could also not respond in time, i.e. the procedure repeats). If an expert finally sends an expertise, it is received by the director and forwarded to the referee. The referee files the results containing the patient interviews as well as the expertise and afterward creates a report. While the referee is doing this, the manager fills a check to pay the expenses of the expert."

# 此函数用于模拟使用Llama模型处理单句并提取实体和关系的过程
# 在真实场景中，您需要根据Llama模型的具体能力来实现这个函数
def simulate_llama_processing(sentence: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    # 假设Llama模型返回了以下实体和关系
    entities = ["referee", "quality assurance department", "managing director", "hospital"]
    relations = [
        ("referee", "member_of", "quality assurance department"),
        ("managing director", "works_for", "hospital"),
    ]
    return entities, relations

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    description: str,
    title: str,
    max_seq_len: int = 512,
    max_gen_len: int = 128,
    max_batch_size: int = 4,
    temperature: float = 0.6,
    top_p: float = 0.9,
):
    """
    Processes a business process description to create a knowledge graph.
    
    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        description (str): The business process description text.
        title (str): The title of the business process.
        Other arguments control how the Llama model generates text.
    """
    # 初始化Llama模型
    llama_model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # 分割描述到单个句子
    sentences = description.split('. ')

    # 初始化知识图谱
    G = nx.DiGraph()
    
    # 为图添加标题节点
    G.add_node(title)

    for sentence in sentences:
        # 使用Llama模型处理每个句子
        # 实际应用中您需要用Llama模型来提取实体和关系
        entities, relations = simulate_llama_processing(sentence)
        
        # 添加实体到图中
        for entity in entities:
            G.add_node(entity)
            G.add_edge(title, entity)  # 将实体与标题连接
        
        # 添加关系到图中
        for source, relation, target in relations:
            if source in G and target in G:
                G.add_edge(source, target, label=relation)
    
    # 可视化知识图谱
    pos = nx.spring_layout(G)  # 使用spring布局
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title('Knowledge Graph: ' + title)
    # plt.show()
    plt.savefig('../images/graph2.png')


if __name__ == "__main__":
    # 使用命令行参数运行程序，或者直接调用main函数
    fire.Fire(main)
