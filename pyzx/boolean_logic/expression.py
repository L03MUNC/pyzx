import numpy as np

from ..utils import EdgeType, VertexType, get_h_box_label, set_h_box_label
from ..graph.graph_s import GraphS
from ..simplify import to_gh, full_reduce
from ..hsimplify import zh_simp, had_edge_to_hbox_simp
from ..extract import extract_circuit


class LogicExpressionGraph(GraphS):
    """Graph representation for boolean-logic expressions.

    This backend extends :class:`GraphS` with helpers for ZH-form checks,
    simplification, and circuit extraction.
    """

    backend = 'logic-expression'


    def __init__(self, graph=None):
        """Initialize a logic-expression graph.

        Parameters
        ----------
        graph : GraphS, optional
            Graph whose structure is merged into this instance.
        """

        super().__init__()
        if graph:
            self @= graph


    # Override reflective operators to prioritize LogicExpressionGraph instances when combined
    # with instances of superclasses

    def __radd__(self, other):
        return super().__radd__(other)

    def __rmul__(self, other):
        return super().__rmul__(other)

    def __rmatmul__(self, other):
        return super().__rmatmul__(other)


    def to_circuit(self, *args, **kwargs):
        """Extract a circuit from the logic-expression graph.

        Parameters
        ----------
        *args
            Positional arguments forwarded to :func:`extract_circuit`.
        **kwargs
            Keyword arguments forwarded to :func:`extract_circuit`.

        Returns
        -------
        Circuit
            Circuit extracted from a prepared copy of the graph.
        """

        g = self.copy()
        outputs = list(g.outputs())
        outp_row = g.row(outputs[0])
        lowest_qubit = min(
            {
                g.qubit(v)
                for v in g.vertex_set()
                if g.row(v) >= outp_row - 2 and g.row(v) <= outp_row
            }
        )

        for i in range(1, g.num_inputs()):
            z = g.add_vertex(VertexType.Z, lowest_qubit-i, outp_row-2)
            h = g.add_vertex(VertexType.H_BOX, lowest_qubit-i, outp_row-1)
            outp = g.add_vertex(VertexType.BOUNDARY, lowest_qubit-i, outp_row)
            outputs.append(outp)
            g.add_edges([(z, h), (h, outp)])
        g.set_outputs(outputs)
        res = extract_circuit(g, *args, **kwargs)
        return res


    def simplify(self):
        """Simplify the graph with ZH rewrite rules.

        Returns
        -------
        LogicExpressionGraph
            The simplified graph instance.

        Raises
        ------
        ValueError
            If the graph is not in ZH form.
        """

        if not self.is_zh():
            raise ValueError('Graph must be a ZH-diagram to be simplified. Use to_zh() to convert '
                'it first.')

        zh_simp(self)
        had_edge_to_hbox_simp(self)
        return self


    def is_zh(self):
        """Check whether the graph is in ZH form.

        Returns
        -------
        bool
            ``True`` if all vertices and edges match the ZH constraints.
        """

        vtypes = set(self.types().values())
        etypes = {self.edge_type(e) for e in self.edges()}
        return vtypes.issubset(
            {VertexType.BOUNDARY, VertexType.Z, VertexType.H_BOX}
        ) and etypes.issubset(
            {EdgeType.SIMPLE}
        )


    def to_zh(self):
        """Convert a copy of the graph to ZH form.

        Returns
        -------
        LogicExpressionGraph
            Converted graph in ZH-compatible form.
        """

        g = self.copy()
        to_gh(g)
        had_edge_to_hbox_simp(g)
        return g


    @staticmethod
    def validate(graph):
        """Validate whether a graph can be used as a logic expression.

        Parameters
        ----------
        graph : GraphS
            Graph to validate.

        Returns
        -------
        bool
            ``True`` if the graph has exactly one output.
        """

        return graph.num_outputs() == 1
