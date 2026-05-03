import re, itertools

import numpy as np
from fuzzingbook.Grammars import Grammar

from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean, BooleanFalse, BooleanTrue, Not, And, Or, Xor
from sympy.parsing.sympy_parser import parse_expr, auto_symbol


class SymPyBooleanExpression:
    """Wrap a SymPy boolean expression.

    Parameters
    ----------
    expr : sympy.logic.boolalg.Boolean or sympy.core.symbol.Symbol
        Expression to wrap.
    """


    EBNF_GRAMMAR_TEMPLATE_PREFIX: Grammar = {
        '<start>': ['<expr>'],
        '<expr>': ['<func>', '<var>'],
        '<func>': ['<unary_func>', '<nary_func>'],
        '<unary_func>': ['Not(<expr>)'],
        '<nary_func>': ['<nary_op>(<expr>(, <expr>)+)'],
        '<nary_op>': ['And', 'Or', 'Xor'],
    }
    """EBNF grammar generting the supported boolean expression syntax with
    operators in prefix notation.

    Notes
    -----
    ``fuzzingbook.Grammars.is_valid_grammar`` returns False as the grammar
    must be extended with a production for the ``<var>`` nonterminal producing
    the desired number of variables."""

    EBNF_GRAMMAR_TEMPLATE_INFIX: Grammar = {
        '<start>': ['<expr>'],
        '<expr>': ['<func>', '<var>'],
        '<func>': ['<unary_func>', '<nary_func>'],
        '<unary_func>': ['~(<expr>)'],
        '<nary_func>': ['<and>', '<or>', '<xor>'],
        '<and>': ['(<expr>( & <expr>)+)'],
        '<or>': ['(<expr>( | <expr>)+)'],
        '<xor>': ['(<expr>( ^ <expr>)+)'],
    }
    """EBNF grammar generting the supported boolean expression syntax with
    operators in infix notation.

    Notes
    -----
    ``fuzzingbook.Grammars.is_valid_grammar`` returns False as the grammar
    must be extended with a production for the ``<var>`` nonterminal producing
    the desired number of variables."""


    def __init__(self, expr):
        """Initialize the expression wrapper.

        Parameters
        ----------
        expr : sympy.logic.boolalg.Boolean or sympy.core.symbol.Symbol
            Expression to wrap.

        Raises
        ------
        TypeError
            If ``expr`` is not a SymPy boolean expression or symbol.
        """

        if not isinstance(expr, (Boolean, Symbol)):
            raise TypeError('Input must be a sympy boolean expression or symbol, got '
                f'{type(expr)}')
        self._expr = expr


    @property
    def expr(self):
        """Return the wrapped SymPy expression.

        Returns
        -------
        sympy.logic.boolalg.Boolean or sympy.core.symbol.Symbol
            Wrapped expression.
        """

        return self._expr

    @property
    def vars(self):
        """Return symbols used in the expression.

        Returns
        -------
        list[sympy.core.symbol.Symbol]
            Free symbols sorted by name.
        """

        return sorted(self._expr.free_symbols, key=lambda s: s.name)


    def __repr__(self):
        return f'SymPyBooleanExpression({repr(self._expr)})'

    def __str__(self):
        return str(self._expr)

    def __call__(self, *args, **kwargs):
        return self._expr.subs(*args, **kwargs)


    @classmethod
    def from_string(cls, string, simplify=False):
        """Convert a string representation of a boolean expression to a SymPy expression.

        Parameters
        ----------
        string : str
            Boolean expression using symbolic operators or aliases.

        Returns
        -------
        sympy.logic.boolalg.Boolean or sympy.core.symbol.Symbol
            SymPy expression equivalent of the parsed string.

        Raises
        ------
        TypeError
            If ``string`` is not of type ``str``.
        ValueError
            If invalid characters are present or parsing fails.

        Example
        -------
        >>> from pyzx.boolean_logic import SymPyBooleanExpression
        >>> expr = SymPyBooleanExpression.from_string('x_1 * -(x_2 + x_3 + x_4) ^ x_3')
        >>> print(expr)
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
        character_whitelist = r'\w\s~&\|\^\(\)\[\]\{\}=<>,'
        if match := re.findall(f'[^{character_whitelist}]', string):
            raise ValueError('String contains invalid characters: ' + ', '.join(set(match)))

        try:
            expr = parse_expr(string, transformations=(auto_symbol,), evaluate=not simplify)
        except (ValueError, TypeError, SyntaxError) as e:
            raise ValueError(f'Error parsing expression. Parser got input: "{string}"\n{e}')
        return cls(expr)


    def to_matrix(self):
        """Convert the expression to a matrix.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(2**n, 2)`` where ``n`` is the number of variables
            in the expression.
        """

        return np.array([
            [0, 1] if self(zip(self.vars, assignment)) else [1, 0]
            for assignment in itertools.product([False, True], repeat=len(self.vars))
        ], dtype=np.complex128).T


    def gate_counts(self):
        """Count occurrences of boolean operators in the expression.

        Returns
        -------
        dict[str, int]
            Mapping from SymPy node type name (e.g., ``'And'``, ``'Or'``,
            ``'Not'``, ``'Xor'``) to the number of occurrences in the
            expression tree.
        """

        def count_gates(expr, dictionary):
            if expr.args:
                dictionary[expr.func.__name__] = dictionary.get(expr.func.__name__, 0) + 1
                for arg in expr.args:
                    count_gates(arg, dictionary)

        counts = dict()
        count_gates(self._expr, counts)
        return counts


    def depth(self):
        """Compute the expression tree depth.

        Returns
        -------
        int
            Maximum nesting depth of boolean operators. Leaf expressions
            (symbols/constants) have depth 0.
        """

        def count_depth(expr, current_depth=0):
            if not expr.args:
                return current_depth
            return max(count_depth(arg, current_depth + 1) for arg in expr.args)

        return count_depth(self._expr)

