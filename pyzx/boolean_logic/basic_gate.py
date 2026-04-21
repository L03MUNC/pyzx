import re
from functools import reduce

from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean, BooleanFalse, BooleanTrue, Not, And, Or, Xor
from sympy.parsing.sympy_parser import parse_expr as sympy_parse_expr

from ..utils import VertexType, get_h_box_label, set_h_box_label
from .expression import LogicExpressionGraph


class LogicGateGraph(LogicExpressionGraph):
    """Graph representation of boolean gates and gate compositions."""

    def __init__(self, graph=None):
        super().__init__(graph)


    @classmethod
    def from_string(cls, string):
        """Parse a boolean expression string into a gate graph.

        Parameters
        ----------
        string : str
            Boolean expression using symbolic operators or aliases.

        Returns
        -------
        LogicGateGraph
            Graph equivalent of the parsed expression.

        Raises
        ------
        TypeError
            If ``string`` is not of type ``str``.
        ValueError
            If invalid characters are present or parsing fails.

        Example
        -------
        >>> from pyzx.boolean_logic import LogicGateGraph
        >>> from pyzx import draw
        >>> g = LogicGateGraph.from_string('x_1 * -(x_2 + x_3 + x_4) ^ x_3')
        >>> draw(g)
        """

        if not isinstance(string, str):
            raise TypeError(f'Input must be a string, got {type(string)}')

        # Unify alternative operator representations
        alt_operators = {
            r'(?<![\w])0(?![\w])': 'false',
            r'(?<![\w])1(?![\w])': 'true',
            r'(?<![\w])not(?![\w])': '~',
            '-': '~',
            r'(?<![\w])and(?![\w])': '&',
            r'\*': '&',
            r'(?<![\w])or(?![\w])': '|',
            r'\+': '|',
            r'(?<![\w])xor(?![\w])': '^',
        }
        for op in alt_operators:
            string = re.sub(op, alt_operators[op], string)

        # Filter input string for valid characters because sympy_parse_expr uses eval()
        character_whitelist = r'\w\s~&\|\^\(\)\[\]\{\}=<>'
        if match := re.findall(f'[^{character_whitelist}]', string):
            raise ValueError('String contains invalid characters: ' + ', '.join(match))

        try:
            expr = sympy_parse_expr(string)
        except (ValueError, TypeError, SyntaxError) as e:
            raise ValueError(f'Error parsing expression: {e}\nParser got input: "{string}"')
        return cls.from_sympy(expr)


    @classmethod
    def from_sympy(cls, expr):
        """Build a gate graph from a SymPy boolean expression.

        Parameters
        ----------
        expr : sympy.logic.boolalg.Boolean or sympy.core.symbol.Symbol
            SymPy expression to convert.

        Returns
        -------
        LogicGateGraph
            Graph equivalent of ``expr``.

        Raises
        ------
        TypeError
            If ``expr`` is not a supported SymPy boolean type.
        ValueError
            If ``expr`` contains unsupported SymPy functions.
        """

        if not isinstance(expr, (Boolean, Symbol)):
            raise TypeError('Input must be a sympy boolean expression or symbol, got '
                f'{type(expr)}')

        gate_map = {
            BooleanFalse: lambda args: ConstantGateGraph(False),
            BooleanTrue: lambda args: ConstantGateGraph(True),
            Not: lambda args: NotGateGraph(),
            And: lambda args: AndGateGraph(len(args)),
            Or: lambda args: OrGateGraph(len(args)),
            Xor: lambda args: XorGateGraph(len(args)),
        }

        symbol_graph = LogicGateGraph()
        inp = symbol_graph.add_vertex(VertexType.BOUNDARY, 0, 0)
        copy_spider = symbol_graph.add_vertex(VertexType.Z, 0, 1)
        outp = symbol_graph.add_vertex(VertexType.BOUNDARY, 0, 2)
        symbol_graph.add_edges([(inp, copy_spider), (copy_spider, outp)])
        symbol_graph.set_inputs((inp,))
        symbol_graph.set_outputs((outp,))

        def _construct_gate(expr):
            func = expr.func
            if func in gate_map:
                args_expr = [_construct_gate(arg) for arg in expr.args] + [LogicGateGraph()]
                args_graph = reduce(lambda x, y: x @ y, args_expr)
                func_graph = gate_map[func](expr.args)
                return func_graph * args_graph
            elif func is Symbol:
                # Create copy spider associated with expression symbol
                g = symbol_graph.copy()
                g.set_vdata(copy_spider, 'symbol', expr.name)
                return g
            else:
                raise ValueError(f'Unsupported sympy function: {func}')

        g = _construct_gate(expr)
        # Connect all copy spiders associated with the same symbol
        symbol_inputs, remove_vertices = dict(), list()
        for v in g.vertices():
            symbol = g.vdata(v, 'symbol')
            if symbol:
                if symbol in symbol_inputs:
                    neighbor0, neighbor1 = list(g.neighbors(v))
                    inp, other = (neighbor0, neighbor1) \
                        if g.type(neighbor0) == VertexType.BOUNDARY else (neighbor1, neighbor0)
                    g.add_edge((symbol_inputs[symbol], other))
                    remove_vertices += [v, inp]
                else:
                    symbol_inputs[symbol] = v
        g.remove_vertices(remove_vertices)
        return g


class ConstantGateGraph(LogicGateGraph):
    """Single-output graph for a boolean constant."""

    def __init__(self, value):
        """Initialize a constant gate.

        Parameters
        ----------
        value : bool
            Constant output value.

        Raises
        ------
        TypeError
            If ``value`` is not a boolean.
        """

        if not isinstance(value, bool):
            raise TypeError(f'Constant gate value must be a boolean, got {type(value)}')

        super().__init__()
        z = self.add_vertex(VertexType.Z, 0, 0, int(value))
        h = self.add_vertex(VertexType.H_BOX, 0, 1)
        outp = self.add_vertex(VertexType.BOUNDARY, 0, 2)
        self.add_edges([(z, h), (h, outp)])
        self.set_outputs((outp,))


class IdentityGateGraph(LogicGateGraph):
    """Single-input identity gate graph."""

    def __init__(self):
        """Initialize an identity gate with one input and one output."""

        super().__init__()
        inp = self.add_vertex(VertexType.BOUNDARY, 0, 0)
        outp = self.add_vertex(VertexType.BOUNDARY, 0, 1)
        self.add_edge((inp, outp))
        self.set_inputs((inp,))
        self.set_outputs((outp,))


class NotGateGraph(LogicGateGraph):
    """Single-input NOT gate graph."""

    def __init__(self):
        """Initialize a NOT gate with one input and one output."""

        super().__init__()
        inp = self.add_vertex(VertexType.BOUNDARY, 0, 0)
        h = self.add_vertex(VertexType.H_BOX, 0, 1)
        z = self.add_vertex(VertexType.Z, 0, 2, 1)
        h2 = self.add_vertex(VertexType.H_BOX, 0, 3)
        outp = self.add_vertex(VertexType.BOUNDARY, 0, 4)
        self.add_edges([(inp, h), (h, z), (z, h2), (h2, outp)])
        self.set_inputs((inp,))
        self.set_outputs((outp,))


class AndGateGraph(LogicGateGraph):
    """Multi-input AND gate graph."""

    def __init__(self, num_inputs=2):
        """Initialize an AND gate.

        Parameters
        ----------
        num_inputs : int, default=2
            Number of input wires.

        Raises
        ------
        ValueError
            If ``num_inputs`` is smaller than 1.
        """

        if num_inputs < 1:
            raise ValueError(f'AND gate must have at least 1 input, got {num_inputs}')

        super().__init__()
        h1 = self.add_vertex(VertexType.H_BOX, num_inputs-1, 1)
        h2 = self.add_vertex(VertexType.H_BOX, num_inputs-1, 2)
        outp = self.add_vertex(VertexType.BOUNDARY, num_inputs-1, 3)
        self.add_edges([(h1, h2), (h2, outp)])
        inputs = list()
        for i in range(num_inputs):
            inp = self.add_vertex(VertexType.BOUNDARY, i, 0)
            inputs.append(inp)
            self.add_edge((inp, h1))
        self.set_inputs(inputs)
        self.set_outputs((outp,))


class OrGateGraph(LogicGateGraph):
    """Multi-input OR gate graph."""

    def __init__(self, num_inputs=2):
        """Initialize an OR gate.

        Parameters
        ----------
        num_inputs : int, default=2
            Number of input wires.

        Raises
        ------
        ValueError
            If ``num_inputs`` is smaller than 1.
        """

        if num_inputs < 1:
            raise ValueError(f'OR gate must have at least 1 input, got {num_inputs}')

        super().__init__()
        h3 = self.add_vertex(VertexType.H_BOX, num_inputs-1, 4)
        z = self.add_vertex(VertexType.Z, num_inputs-1, 5, 1)
        h4 = self.add_vertex(VertexType.H_BOX, num_inputs-1, 6)
        outp = self.add_vertex(VertexType.BOUNDARY, num_inputs-1, 7)
        self.add_edges([(h3, z), (z, h4), (h4, outp)])
        inputs = list()
        for i in range(num_inputs):
            inp = self.add_vertex(VertexType.BOUNDARY, i, 0)
            h1 = self.add_vertex(VertexType.H_BOX, i, 1)
            z = self.add_vertex(VertexType.Z, i, 2, 1)
            h2 = self.add_vertex(VertexType.H_BOX, i, 3)
            inputs.append(inp)
            self.add_edges([(inp, h1), (h1, z), (z, h2), (h2, h3)])
        self.set_inputs(inputs)
        self.set_outputs((outp,))


class XorGateGraph(LogicGateGraph):
    """Multi-input XOR gate graph."""

    def __init__(self, num_inputs=2):
        """Initialize an XOR gate.

        Parameters
        ----------
        num_inputs : int, default=2
            Number of input wires.

        Raises
        ------
        ValueError
            If ``num_inputs`` is smaller than 1.
        """

        if num_inputs < 1:
            raise ValueError(f'XOR gate must have at least 1 input, got {num_inputs}')

        super().__init__()
        z = self.add_vertex(VertexType.Z, num_inputs-1, 2)
        h = self.add_vertex(VertexType.H_BOX, num_inputs-1, 3)
        outp = self.add_vertex(VertexType.BOUNDARY, num_inputs-1, 4)
        self.add_edges([(z, h), (h, outp)])
        inputs = list()
        for i in range(num_inputs):
            inp = self.add_vertex(VertexType.BOUNDARY, i, 0)
            h = self.add_vertex(VertexType.H_BOX, i, 1)
            inputs.append(inp)
            self.add_edges([(inp, h), (h, z)])
        self.set_inputs(inputs)
        self.set_outputs((outp,))
