# copyright ZJX

import math
import matplotlib.pyplot as plt
import networkx as nx
import queue
import sys

from lxml import etree
from pathlib import Path

NodeType = etree._Element
plt.switch_backend('Agg')


xml_file = Path(sys.argv[1])
doc = etree.parse(xml_file, etree.get_default_parser())


def get_node_uid(node:NodeType):
    if node is None:
        return ''
    elif node.tag == 'cpu':
        return f"cpu_{node.get('numaid', -1)}"
    elif node.tag == 'gpu':
        return f"gpu_{node.get('dev', -1)}_rank_{node.get('rank', '-1')}"
    elif node.tag == 'pci':
        return f"pci_{node.get('busid', 'null')}"
    elif node.tag == 'net':
        return f"net_{node.get('name', 'null')}"
    else:
        p_uid = get_node_uid(node.getparent())
        return p_uid + '/' + node.tag


def convert_xml_to_digraph(tree:NodeType):
    graph = nx.DiGraph()
    ckpt:queue.Queue[NodeType] = queue.Queue()
    ckpt.put(tree)

    while not ckpt.empty():
        parent = ckpt.get()
        p_uid = get_node_uid(parent)

        if not graph.has_node(p_uid):
            graph.add_node(p_uid, color='green')

        for child in parent.getchildren():
            child:NodeType = child
            ckpt.put(child)
            c_uid = get_node_uid(child)
            graph.add_edge(p_uid, c_uid, label=f'', edge_width='5')

            # connect nvlink to another pci device
            # if child.tag == 'nvlink':
            #     target_uid = 'pci_' + child.get('target', 'null')
            #     graph.add_edge(c_uid, target_uid, label=f'nvlink', edge_width='10')

    return graph


def save_to_svg(graph: nx.DiGraph):
    fig, ax = plt.subplots()
    FIG_WIDTH=8
    fig.set_size_inches(FIG_WIDTH * 2,  FIG_WIDTH)

    ax.set_title(f"System topo")
    # 分成两张图绘制，一个是 pci 连接拓扑，另一个是 nvlink 连接拓扑
    pos = nx.spring_layout(graph, k= 4/math.sqrt(graph.number_of_nodes()))
    nx.draw(graph, pos, ax=ax, with_labels=True,
            arrowsize=20, edge_color='green',
            font_weight='bold', font_size=8,
            node_size=600)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, ax=ax,
                                edge_labels=edge_labels,
                                font_size=6,
                                label_pos=0.3)

    output_file = xml_file.with_suffix(".png")
    print("Save file to:", output_file)
    plt.savefig(output_file)


graph_data = convert_xml_to_digraph(doc.getroot())
save_to_svg(graph_data)
