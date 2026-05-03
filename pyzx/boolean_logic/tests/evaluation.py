import sys, random, json
from enum import Enum
from time import perf_counter
from pathlib import Path

import numpy as np
from fuzzingbook.Grammars import extend_grammar, convert_ebnf_grammar
from fuzzingbook.GrammarCoverageFuzzer import GrammarCoverageFuzzer, duplicate_context

from ..gate_graph import LogicGateGraph
from ..expression import SymPyBooleanExpression
from . import helper as h


class Evaluation:
    """Manage and run a population of test cases.

    Notes
    -----
    Instances collect :class:`TestCase` objects in ``population`` and store
    runtime metadata in ``metadata``.
    """

    def __init__(self):
        """Initialize an empty evaluation."""

        self.population = list()
        self._metadata = dict()


    @property
    def metadata(self):
        return self._metadata


    def add_test_cases(self, test_cases, unique=False, variable_names=None):
        """Add one or more test cases to the population.

        Parameters
        ----------
        test_cases : str or TestCase or list[str | TestCase]
            Test case(s) to add. Strings are converted to :class:`TestCase`.
        unique : bool, optional
            If True, only add cases whose expression is *structurally unique*
            up to variable renaming. Uniqueness is determined by replacing each
            variable name in ``variable_names`` with the placeholder ``'x'`` in
            the canonicalized expression string.
        variable_names : list[str] or None, optional
            Variable names used for the uniqueness check when ``unique`` is
            True. Must be provided if ``unique`` is True.

        Raises
        ------
        ValueError
            If ``unique`` is True and ``variable_names`` is None.
        TypeError
            If an element of ``test_cases`` is neither ``str`` nor
            :class:`TestCase`.
        """

        if unique and variable_names is None:
            raise ValueError('variable_names must be provided when unique is True')

        def unique_expr_string(tc, variable_names):
            result = tc.expr_string
            for var in variable_names:
                result = result.replace(var, 'x')
            return result

        if unique:
            existing = {unique_expr_string(tc, variable_names) for tc in self.population}
        if isinstance(test_cases, (str, TestCase)):
            test_cases = [test_cases]
        for tc in test_cases:
            if isinstance(tc, str):
                tc = TestCase(tc)
            elif not isinstance(tc, TestCase):
                raise TypeError(f'Expected str or TestCase or list of these, got {type(tc)}')
            if unique: 
                if (string := unique_expr_string(tc, variable_names)) in existing:
                    continue
                else:
                    existing.add(string)
            self.population.append(tc)


    def add_metadata(self, **kwargs):
        """Add key-value pairs to the evaluation metadata.

        Parameters
        ----------
        **kwargs
            Arbitrary key-value pairs to add to ``metadata``.
        """

        self._metadata.update(kwargs)


    def gen_population(self, num_cases, num_vars, grammar, seed=None, verbosity=0):
        """Generate a population of random boolean-expression test cases.

        Parameters
        ----------
        num_cases : int
            Target number of test cases to generate.
        num_vars : int
            Number of distinct variables to use.
        grammar : dict
            EBNF grammar template describing the expression language. It must
            contain a ``<var>`` nonterminal which will be replaced by the
            generated variable names.
        seed : int or None, optional
            If given, seed Python's ``random`` module for reproducibility.
        verbosity : int, optional
            Verbosity level forwarded to the logger.

        Raises
        ------
        ValueError
            If the derived grammar cannot be used to construct the fuzzer.

        Notes
        -----
        The generated test cases are added to :attr:`population` with
        ``unique=True`` (up to variable renaming). This method also updates
        :attr:`metadata` with ``num_cases``, ``num_vars`` and ``seed``.
        """

        def gen_fuzzer(grammar):
            try:
                return GrammarCoverageFuzzer(grammar)
            except AssertionError as e:
                raise ValueError('Error generating GrammarCoverageFuzzer. Probably, the grammar '
                    f'is invalid:\n{grammar}') from e

        h.printv(f'Generating population with {num_cases} cases and {num_vars} variables',
            verbosity=verbosity, level=1)
        if num_cases > 2 ** num_vars:
            h.printv(f'Warning: {num_cases=} exceeds the number of possible boolean functions for '
                f'{num_vars=}', verbosity=verbosity, level=0)

        if seed:
            random.seed(seed)
        self._metadata.update({'num_cases': num_cases, 'num_vars': num_vars, 'seed': seed})
        h.printv('\tGenerating grammar in BNF form', verbosity=verbosity, level=2)
        variable_names = [f'x_{i}' for i in range(num_vars)]
        ebnf_grammar = extend_grammar(grammar, {'<var>': variable_names})
        bnf_grammar = convert_ebnf_grammar(ebnf_grammar)

        fuzzer = gen_fuzzer(bnf_grammar)
        num_fuzzes = 0
        while len(self.population) < num_cases:
            if not fuzzer.missing_expansion_coverage():
                symbol = '<func>'
                h.printv(f'\tAll expansions covered, duplicating context of {symbol}',
                    verbosity=verbosity, level=2)
                duplicate_context(bnf_grammar, symbol)
            self.add_test_cases(fuzzer.fuzz(), True, variable_names)
            num_fuzzes += 1
        h.printv(f'Generated {num_cases} test cases in {num_fuzzes} fuzzes, '
            f'{(100*num_cases/num_fuzzes):.0f}% efficiency', verbosity=verbosity, level=1)


    def print_population(self, **kwargs):
        """Pretty-print the current population.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :func:`print`. If ``file`` is not
            provided, it defaults to ``sys.stderr``.
        """

        kwargs['file'] = kwargs.get('file') or sys.stderr
        print('[', *[f'\'{tc.expr_string}\'' for tc in self.population], ']',
            sep='\n', **kwargs)


    def run(self, verbosity=0, callback=None):
        """Run all test cases in the population.

        Parameters
        ----------
        verbosity : int, optional
            Verbosity level forwarded to each test case.
        callback : callable, optional
            Optional hook invoked after each executed test case, with signature
            ``callback(index, test_case)`` where:

            - ``index`` (int) is the 0-based index in ``population``.
            - ``test_case`` (TestCase) is the executed instance. Its
                :attr:`~TestCase.result` has been set.

        Notes
        -----
        Stores total runtime in ``metadata['runtime_ms']``.
        """

        h.printv(f'Running evaluation with {len(self.population)} test cases',
            verbosity=verbosity, level=1)
        t0 = perf_counter()
        try:
            for i, tc in enumerate(self.population):
                tc(verbosity)
                if callback:
                    callback(i, tc)
        finally:
            self._metadata['runtime_ms'] = int((perf_counter() - t0) * 1000)
        h.printv(f'Evaluation completed in {self._metadata["runtime_ms"]} ms\n',
            verbosity=verbosity, level=1)


    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


    def result_summary(self, filepath=None):
        """Build and optionally persist an aggregated summary of all test cases.

        Parameters
        ----------
        filepath : str or pathlib.Path, optional
            If given, write the summary dictionary as JSON to this path.

        Returns
        -------
        dict
            A dictionary with the following structure::

                {
                    'metadata': dict,
                    'result_enum': list[str],
                    'test_cases': list[dict],
                    'result_counts': list[int],
                    'to_result_counts': dict[str, dict[int, list[int]]],
                    'from_result_counts': dict[str, list[dict[int, int]]],
                }

            with the keys defined as follows.

            ``metadata``
                The evaluation metadata dictionary (e.g. may contain
                ``'runtime_ms'`` after :meth:`run`).

            ``result_enum``
                List of :class:`TestResult` names in enum definition order.

            ``test_cases``
                List of per-test-case dictionaries as returned by
                :meth:`TestCase.to_dict`, i.e.::

                    {'expr_string': <str>, 'result': <TestResult name>}

            ``result_counts``
                Counts per :class:`TestResult` value.

            ``to_result_counts``
                Nested mapping ``metric -> metric_value -> counts`` where
                ``counts`` is a list with one entry per :class:`TestResult` value.

            ``from_result_counts``
                Mapping ``metric -> counts_by_result`` where
                ``counts_by_result`` is a list of dictionaries, one per
                :class:`TestResult` value, mapping ``metric_value -> count``.
        """

        # Minimum and maximum TestResult values for indexing result counts
        list_index_min, list_index_max = -2, 3
        metrics = [
            'num_vars', 'num_gates', 'num_not_gates', 'num_and_gates', 'num_or_gates',
            'num_xor_gates', 'expr_depth',
        ]
        lod = [tc.summary() for tc in self.population]

        result_counts = [0] * (list_index_max - list_index_min + 1)
        for tc in lod:
            result_counts[tc['result'] - list_index_min] += 1

        summary = {
            'metadata': self._metadata,
            'result_enum': [val.name for val in TestResult],
            'test_cases': [tc.to_dict() for tc in self.population],
            'result_counts': result_counts,
            # Structure of 'to_result_counts': {
            #   '<metric_0>': {
            #       <key_0>: [<num_invalid>, <num_undefined>, ...],
            #       <key_1>: [<num_invalid>, <num_undefined>, ...],
            #   ...},
            # ...}
            'to_result_counts': h.dodol_from_lod(
                lod, metrics, 'result', (list_index_min, list_index_max)
            ),
            # Structure of 'from_result_counts': {
            #   '<metric_0>': [
            #       {<key_0>: <num_invalid>, <key_1>: <num_invalid>, ...},
            #       {<key_0>: <num_undefined>, <key_1>: <num_undefined>, ...},
            #   ...],
            # ...}
            'from_result_counts': h.dolod_from_lod(
                lod, metrics, 'result', (list_index_min, list_index_max)
            ),
        }

        if filepath:
            with open(Path(filepath), 'w') as f:
                json.dump(summary, f, indent=4)

        return summary


class TestCase:
    """Single end-to-end evaluation of ZH-tools for boolean expressions.

    Parameters
    ----------
    expr_string : str
        Input expression in the supported parser syntax.

    Notes
    -----
    The test pipeline currently builds a ZH-diagram and simplifies it, verifying
    matrix equivalence after each step.
    """

    def __init__(self, expr_string):
        """Parse and precompute representations for a test case.

        Parameters
        ----------
        expr_string : str
            Boolean expression string.
        """

        self._expr_sympy = SymPyBooleanExpression.from_string(expr_string)
        self._expr_string = str(self._expr_sympy)
        self._expr_matrix = self._expr_sympy.to_matrix()
        self._zh = self._zh_matrix = self._simplified = self._simplified_matrix = self._circuit \
            = self._circuit_matrix = None
        self._result = TestResult.UNDEFINED


    @property
    def expr_string(self):
        """str: Canonicalized expression string."""

        return self._expr_string

    @property
    def expr_sympy(self):
        """SymPyBooleanExpression: Parsed SymPy expression wrapper."""

        return self._expr_sympy

    @property
    def expr_matrix(self):
        """numpy.ndarray: Matrix representation of the expression."""

        return self._expr_matrix

    @property
    def zh(self):
        """LogicGateGraph or None: ZH-diagram representation (if created)."""

        return self._zh

    @property
    def zh_matrix(self):
        """numpy.ndarray or None: Matrix of the ZH-diagram (if created)."""

        return self._zh_matrix

    @property
    def simplified(self):
        """LogicExpressionGraph or None: Simplified diagram (if created)."""

        return self._simplified

    @property
    def simplified_matrix(self):
        """numpy.ndarray or None: Matrix of the simplified diagram (if created)."""

        return self._simplified_matrix

    @property
    def circuit(self):
        """Circuit or None: Extracted quantum circuit (if created)."""

        return self._circuit

    @property
    def circuit_matrix(self):
        """numpy.ndarray or None: Matrix of the extracted quantum circuit (if created)."""

        return self._circuit_matrix

    @property
    def result(self):
        """TestResult: Outcome of the test case.

        Notes
        -----
        The value is :attr:`TestResult.UNDEFINED` until :meth:`run` is called.
        """

        return self._result


    def __repr__(self):
        return f'TestCase({self._expr_string})'

    def __str__(self):
        return self.__repr__()

    def to_dict(self):
        return {'expr_string': self._expr_string, 'result': self._result.name}


    def create_zh(self):
        """Construct a ZH-diagram from the expression and compute its matrix."""

        self._zh = LogicGateGraph.from_boolean_expression(self._expr_sympy)
        self._zh_matrix = self._zh.to_matrix()

    def assert_zh(self):
        """Assert that the ZH-diagram matrix matches the expression.

        Raises
        ------
        ValueError
            If the ZH-diagram has not been created.
        """

        if not self._zh:
            raise ValueError('ZH-diagram not created yet')

        assert self.compare_matrices(self._expr_matrix, self._zh_matrix)


    def simplify(self):
        """Simplify the ZH-diagram and compute the matrix of the simplified diagram.

        Raises
        ------
        ValueError
            If the ZH-diagram has not been created.
        """

        if not self._zh:
            raise ValueError('ZH-diagram not created yet')

        self._simplified = self._zh.copy().simplify()
        self._simplified_matrix = self._simplified.to_matrix()

    def assert_simplified(self):
        """Assert that the simplified diagram matrix matches the expression.

        Raises
        ------
        ValueError
            If the simplified diagram has not been created.
        """

        if not self._simplified:
            raise ValueError('Simplified diagram not created yet')

        assert self.compare_matrices(self._expr_matrix, self._simplified_matrix)


    def extract_circuit(self):
        """Extract a quantum circuit from the simplified diagram and compute
        the matrix of the extracted circuit.

        Raises
        ------
        ValueError
            If the simplified diagram has not been created.
        """

        if not self._simplified:
            raise ValueError('Simplified diagram not created yet')

        self._circuit = self._simplified.to_circuit()
        self._circuit_matrix = self._circuit.to_matrix()

    def assert_circuit(self):
        """Assert that the circuit matrix matches the expression.

        Raises
        ------
        ValueError
            If the circuit has not been created.
        """

        if not self._circuit:
            raise ValueError('Circuit not created yet')

        assert self.compare_matrices(self._expr_matrix, self._circuit_matrix)


    def run(self, verbosity=0):
        """Execute the evaluation pipeline.

        Parameters
        ----------
        verbosity : int, optional
            Verbosity level controlling logging to stderr.

            - -1: print nothing at all.
            - 0: only errors (exceptions) are printed.
            - 1: additionally print start/end summary per test case.
            - 2: additionally print per-step actions and assertions.

        Returns
        -------
        TestResult
            Result of the run:

            - PASS: all pipeline steps run and all assertions pass.
            - INVALID: an exception occurs during a pipeline action.
            - FAIL1/FAIL2/FAIL3: assertion fails after step 1/2/3 respectively.

            This method does not return UNDEFINED.
        """

        # Contain the steps of the evaluation pipeline as (act, assert) pairs
        pipeline = [
            (self.create_zh, self.assert_zh),
            (self.simplify, self.assert_simplified),
            # (self.extract_circuit, self.assert_circuit),
            # Circuit extraction is not supported yet
        ]

        h.printv(f'Running test case: {self._expr_string}', verbosity=verbosity, level=1)
        self._result = TestResult.PASS
        for i, (act, assert_) in enumerate(pipeline):
            try:
                h.printv(f'\tRunning step {i+1}: {act.__name__}', verbosity=verbosity, level=2)
                act()
            except Exception as e:
                h.printv(f'Error during {act.__name__}:\n{e}', verbosity=verbosity, level=0)
                self._result = TestResult.INVALID
                break
    
            try:
                h.printv('\tAsserting result', verbosity=verbosity, level=2)
                assert_()
            except AssertionError:
                h.printv(f'\tAssertion failed after step {i+1}', verbosity=verbosity, level=2)
                # Determine TestResult enum value based on which step failed
                self._result = TestResult(i + 1)
                break

        h.printv(f'Test result: {self._result.name}', verbosity=verbosity, level=1)
        return self._result

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


    @staticmethod
    def compare_matrices(m1, m2):
        """Compare two matrices up to a global scalar.

        Parameters
        ----------
        m1, m2 : numpy.ndarray
            Matrices to compare.

        Returns
        -------
        bool
            True if matrices are numerically close up to a global scalar.
        """

        msum = np.abs(m1) + np.abs(m2)
        scalar = np.max(msum) / np.max(np.abs(m1))
        return np.allclose(m1, msum, scalar, 0.25)


    def summary(self):
        """Return a numeric summary of this test case.

        Returns
        -------
        dict
            Dictionary with the following keys:

            ``expr_string``
                Canonical expression string.
            ``result``
                Integer :class:`TestResult` value (e.g. ``0`` for PASS).
            ``num_vars``
                Number of variables in the expression.
            ``num_gates``
                Total number of gates in the expression.
            ``num_not_gates``
                Number of ``Not`` gates.
            ``num_and_gates``
                Number of ``And`` gates.
            ``num_or_gates``
                Number of ``Or`` gates.
            ``num_xor_gates``
                Number of ``Xor`` gates.
            ``expr_depth``
                Maximum operator nesting depth.
        """

        gate_counts = self._expr_sympy.gate_counts()
        return {
            'expr_string': self._expr_string,
            'result': self._result.value,
            'num_vars': len(self._expr_sympy.vars),
            'num_gates': sum(gate_counts.values()),
            'num_not_gates': gate_counts.get('Not', 0),
            'num_and_gates': gate_counts.get('And', 0),
            'num_or_gates': gate_counts.get('Or', 0),
            'num_xor_gates': gate_counts.get('Xor', 0),
            'expr_depth': self._expr_sympy.depth(),
        }


class TestResult(Enum):
    INVALID = -2
    UNDEFINED = -1
    PASS = 0
    # Positive integers indicate which step of the pipeline failed,
    # e.g. FAIL1 means the first check failed
    FAIL1 = 1
    FAIL2 = 2
    FAIL3 = 3
