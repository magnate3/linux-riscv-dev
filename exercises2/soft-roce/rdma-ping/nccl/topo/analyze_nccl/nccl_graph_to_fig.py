# copyright ZJX

import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import xml.etree.ElementTree as ET

from collections import namedtuple
from pathlib import Path


plt.switch_backend('Agg')

xml_file = Path(sys.argv[1])
dom = ET.parse(xml_file)


GraphNode = namedtuple('GraphNode', ['attrib', 'channel'])
DevNode = namedtuple('DevNode', ['type', 'dev'])


def convert_xml_to_pynest(root: ET.Element) -> list:
    ret = []
    for g in root.iter('graph'):
        if g.attrib['nchannels'] == '0':
            continue
        _channel = []
        for _ch in g:
            _channel.append([DevNode(cn.tag, cn.attrib['dev']) for cn in _ch])
        ret.append(GraphNode(g.attrib, _channel))
    return ret


def convert_data_to_digraph(data:GraphNode):
    g = nx.DiGraph()

    for i, ch in enumerate(data.channel):
        nodes = []
        for n in ch:
            node_name = f"{n.type}-{n.dev}"
            nodes.append(node_name)
            if g.has_node(node_name):
                continue
            g.add_node(node_name, color='green')
        
        for u, v in zip(nodes[:-1], nodes[1:]):
            g.add_edge(u, v, label=f"ch {i}", edge_width='5')

    return g


def graphnode_attrib_to_string(data:GraphNode):
    return "\n".join([f"{k}: {v}" for k, v in data.attrib.items()])


def save_to_svg(graph_data: list):
    fig, axes = plt.subplots(len(graph_data), 1)
    FIG_WIDTH=8
    fig.set_size_inches(FIG_WIDTH, FIG_WIDTH*len(axes))

    for i, ax in enumerate(axes):
        ax.set_title(f"Graph {i}")
        attrib = graphnode_attrib_to_string(graph_data[i])
        ax.text(-0.1, 0.95, attrib,
                transform=ax.transAxes, ha='left', va='top')
        
        g = convert_data_to_digraph(graph_data[i])
        pos = nx.kamada_kawai_layout(g)
        nx.draw(g, pos, ax=ax, with_labels=True,
                arrowsize=20, edge_color='green',
                font_weight='bold', font_size=8,
                node_size=600)
        edge_labels = nx.get_edge_attributes(g, 'label')
        nx.draw_networkx_edge_labels(g, pos, ax=ax,
                                    edge_labels=edge_labels,
                                    font_size=6,
                                    label_pos=0.3)
    output_file = xml_file.with_suffix(".png")
    print("Save file to:", output_file)
    plt.savefig(output_file)


def save_to_gexf(graph_data:list):
    for i, dat in enumerate(graph_data):
        g = convert_data_to_digraph(dat)
        file_name = xml_file.parent / (xml_file.stem + f"_g{i:04d}.gexf")
        nx.write_gexf(g, file_name)


graph_data = convert_xml_to_pynest(dom.getroot())
save_to_svg(graph_data)
