from diagrams import Diagram, Cluster, Edge
from diagrams.programming import flowchart

graph_attr = {
    'dpi': '100.0',
}

def draw_high_level():
    with Diagram("Elaboration of a piece", show=False, graph_attr=graph_attr, direction='TB'):
        with Cluster('Piece Elaborator'):
            with Cluster('Internal memory'):
                symbol_memory_dict = flowchart.StoredData('symbol memory dict')
            piece_elaborate = flowchart.Action('Piece elaborate (n_steps, memory melody)')

            with Cluster('init attributes'):
                with Cluster('Manual input templates'):
                    melody_templates = flowchart.ManualInput('melody templates')
                    coherence_template = flowchart.ManualInput('coherence template')

                with Cluster('Melody Elaborator'):
                    with Cluster('hyper parameters'):
                        ops = flowchart.Database('operations')
                        policy = flowchart.Database('policy')
                        imitate_policy = flowchart.Database('imitation policy')
                    elaborate = flowchart.Action('Elaborate (n_steps, memory melody)')
                    [ops, policy, imitate_policy] >> Edge() >> elaborate

            [melody_templates,coherence_template ] >> Edge() >> piece_elaborate
            [symbol_memory_dict] << Edge(color='green') >> piece_elaborate
            elaborate >> Edge(style='dotted') >> piece_elaborate

def draw_melody_template():


if __name__ == '__main__':
    draw_high_level()
