import spacy
import networkx as nx
import matplotlib.pyplot as plt

# 加载Spacy的英语模型
nlp = spacy.load("en_core_web_sm")

text = """
A hospital wants to establish a rating workflow for their doctors. To make the workflow reliable two different roles are assigned. The first one is a referee from the newly created quality assurance department while the second one represents the managing director of the hospital. Both roles execute all of their tasks independently from each other. The referee starts a new case regarding a certain doctor by interviewing patients. Since a patient interview workflow is already established, it is simply integrated in the new workflow. Meanwhile, the director asks an external expert to review the work of the doctor under rating. Unfortunately, since the expert only gets a low expenses fee, it can happen that the expert is not responding in time. If that happens, another expert has to be asked (who could also not respond in time, i.e., the procedure repeats). If an expert finally sends an expertise, it is received by the director and forwarded to the referee. The referee files the results containing the patient interviews as well as the expertise and afterward creates a report. While the referee is doing this, the manager fills a check to pay the expenses of the expert.
"""

# 分割文本为句子
doc = nlp(text)
sentences = [sent.text.strip() for sent in doc.sents]

# 初始化知识图谱
G = nx.DiGraph()

def add_entities_to_graph(doc, graph):
    # 添加实体到图中
    for ent in doc.ents:
        graph.add_node(ent.text, type=ent.label_)

def add_edges_to_graph(doc, graph):
    # 添加边到图中，这里简化为添加依赖标签作为边
    for token in doc:
        for child in token.children:
            graph.add_edge(token.head.text, child.text, relation=child.dep_)

# 遍历句子，添加节点和边到图中
for sentence in sentences:
    doc = nlp(sentence)
    add_entities_to_graph(doc, G)
    add_edges_to_graph(doc, G)

# 绘制知识图谱
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=2000, node_color='lightblue', with_labels=True, arrows=True)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
# plt.show()
plt.savefig('../images/graph.png')
