import sys, random, json
from enum import Enum
from time import perf_counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from fuzzingbook.Grammars import extend_grammar, convert_ebnf_grammar
from fuzzingbook.GrammarCoverageFuzzer import GrammarCoverageFuzzer, duplicate_context
from fuzzingbook.Timeout import Timeout

from ..gate_graph import LogicGateGraph
from ..expression import SymPyBooleanExpression
from . import helper as h
from ...graph.base import BaseGraph
from ...drawing import draw_matplotlib, matrix_to_latex as zx_matrix_to_latex


DEFAULT_STEP_TIMEOUT_LIMIT_MS = 10000


class TestResult(Enum):
    INVALID = -2
    UNDEFINED = -1
    PASS = 0
    # Positive integers indicate which step of the pipeline failed,
    # e.g. FAIL1 means the first check failed
    FAIL1 = 1
    FAIL2 = 2
    FAIL3 = 3

TEST_RESULT_MIN = min(res.value for res in TestResult)
TEST_RESULT_MAX = max(res.value for res in TestResult)


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
                return GrammarCoverageFuzzer(
                    grammar,
                    min_nonterminals=1,
                    max_nonterminals=8,
                )
            except AssertionError as e:
                raise ValueError('Error generating GrammarCoverageFuzzer. Probably, the grammar '
                    f'is invalid:\n{grammar}') from e

        h.printv(f'Generating population with {num_cases} cases and {num_vars} variables',
            verbosity=verbosity, level=1)

        if seed:
            random.seed(seed)
        self._metadata.update({'num_cases': num_cases, 'num_vars': num_vars, 'seed': seed})
        h.printv('\tGenerating grammar in BNF form', verbosity=verbosity, level=2)
        variable_names = [f'x_{i}' for i in range(num_vars)]
        ebnf_grammar = extend_grammar(grammar, {'<var>': variable_names})
        bnf_grammar = convert_ebnf_grammar(ebnf_grammar)

        fuzzer = gen_fuzzer(bnf_grammar)
        num_fuzzes = duplicate = 0
        while len(self.population) < num_cases:
            if not fuzzer.missing_expansion_coverage():
                symbol = '<func>'
                # Increase generation depth
                duplicate += 1
                duplicate_context(bnf_grammar, symbol)
                fuzzer.min_nonterminals += 1
                fuzzer.max_nonterminals += 1
                h.printv(
                    f'    All expansions covered. Duplicating context of {symbol} {duplicate} '
                    + ('times' if duplicate != 1 else 'time')
                    + ' and incrementing interval bounds of nonterminals to '
                    f'({fuzzer.min_nonterminals}, {fuzzer.max_nonterminals})', end='',
                    verbosity=verbosity, level=2
                )
            h.printv(f'\rFuzzer generating test case {len(self.population)+1} of {num_cases}...',
                end='', verbosity=verbosity, level=1)
            self.add_test_cases(fuzzer.fuzz(), True, variable_names)
            num_fuzzes += 1
        h.printv(verbosity=verbosity, level=1)
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


    def run(
            self, verbosity=0, callback=None,
            step_timeout_limit_ms=DEFAULT_STEP_TIMEOUT_LIMIT_MS
        ):
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
        step_timeout_limit_ms : int, optional
            Timeout limit in milliseconds for each pipeline step of each test case.
            Defaults to :data:`DEFAULT_STEP_TIMEOUT_LIMIT_MS`.

        Notes
        -----
        Stores total runtime in ``metadata['runtime_ms']``.
        """

        h.printv(f'Running evaluation with {len(self.population)} test cases\n',
            verbosity=verbosity, level=1)
        t0 = perf_counter()
        try:
            for i, tc in enumerate(self.population):
                h.printv(f'({i+1}/{len(self.population)}) ', end='', verbosity=verbosity, level=1)
                tc(verbosity, step_timeout_limit_ms)
                if callback:
                    callback(i, tc)
        finally:
            self._metadata['runtime_ms'] = int((perf_counter() - t0) * 1000)
        h.printv(f'Evaluation completed, total runtime: {self._metadata["runtime_ms"]} ms\n',
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
                    'timing': {
                        'avg_runtimes': list[int],
                        'timeout_fractions': list[float],
                        'result_to_runtimes': list[list[list[int]]],
                        'result_to_avg_runtimes': list[list[int]],
                    },
                    'result_counts': list[int],
                    'result_fractions': list[float],
                    'metric_to_result': dict[str, dict[int, list[int]]],
                    'result_to_metric': dict[str, list[dict[int, int]]],
                }

            with the keys defined as follows.

            ``metadata``
                The evaluation metadata dictionary (e.g. may contain
                ``'runtime_ms'`` after :meth:`run`).

            ``result_enum``
                List of :class:`TestResult` names in enum definition order.

            ``test_cases``
                List of per-test-case dictionaries as returned by
                :meth:`TestCase.to_dict`.

            ``timing``
                Aggregated timing information across all test cases.

                ``timing['avg_runtimes']``
                    Average runtime in milliseconds for each pipeline step.

                ``timing['timeout_fractions']``
                    Fractions of timeouts per :class:`TestResult` bucket.
                    Entries align with ``result_enum``.

                ``timing['result_to_runtimes']``
                    List of runtimes of each step per :class:`TestResult` bucket.

                ``timing['result_to_avg_runtimes']``
                    List of average runtimes of each step per :class:`TestResult`
                    bucket.

            ``result_counts``
                Counts per :class:`TestResult` value.

            ``metric_to_result``
                Nested mapping ``metric -> metric_value -> counts`` where
                ``counts`` is a list with one entry per :class:`TestResult` value.

            ``result_to_metric``
                Mapping ``metric -> counts_by_result`` where
                ``counts_by_result`` is a list of dictionaries, one per
                :class:`TestResult` value, mapping ``metric_value -> count``.

        Raises
        ------
        ValueError
            If the population is empty and a summary cannot be built.
        """

        if not self.population:
            raise ValueError('Population is empty, cannot build summary')

        # Minimum and maximum TestResult values for indexing result counts
        list_len = TEST_RESULT_MAX - TEST_RESULT_MIN + 1
        metrics = [
            'num_vars', 'num_gates', 'num_not_gates', 'num_and_gates', 'num_or_gates',
            'num_xor_gates', 'expr_depth', 'num_literals',
        ]
        lod = [tc.summary() for tc in self.population]

        result_counts = [0] * list_len
        for tc in lod:
            result_counts[tc['result'] - TEST_RESULT_MIN] += 1
        result_fractions = [round(result / len(self.population), 2) for result in result_counts]
        runtimes = [
            [tc.timing['runtimes'][step] for step in sorted(tc.timing['runtimes'])]
            for tc in self.population
        ]
        avg_runtimes = [
            int(runtime) for runtime in np.nanmean([
                runtime + [np.nan] * (max(len(runtime) for runtime in runtimes) - len(runtime))
                for runtime in runtimes
            ], axis=0)
        ]
        timeout_counts = [0] * list_len
        for tc in lod:
            if tc['timeout']:
                timeout_counts[tc['result'] - TEST_RESULT_MIN] += 1
        timeout_fractions = [
            round(count / result_counts[i], 2) if result_counts[i] > 0 else 0.0
            for i, count in enumerate(timeout_counts)
        ]
        result_to_runtimes = [list() for _ in range(list_len)]
        for tc, runtime in zip(lod, runtimes):
            result_to_runtimes[tc['result'] - TEST_RESULT_MIN].append(runtime)
        result_to_avg_runtimes = list()
        for result in result_to_runtimes:
            if result:
                steps = max(len(tc) for tc in result)
                runtimes = [
                    tc + [np.nan] * (steps - len(tc))
                    for tc in result
                ]
                result_to_avg_runtimes.append(
                    np.nanmean(runtimes, axis=0).astype(int).tolist()
                )
            else:
                result_to_avg_runtimes.append(list())

        summary = {
            'metadata': self._metadata,
            'result_enum': [val.name for val in TestResult],
            'test_cases': [tc.to_dict() for tc in self.population],
            'timing': {
                'avg_runtimes': avg_runtimes,
                'timeout_fractions': timeout_fractions,
                'result_to_runtimes': result_to_runtimes,
                'result_to_avg_runtimes': result_to_avg_runtimes,
            },
            'result_counts': result_counts,
            'result_fractions': result_fractions,
            # Structure of 'metric_to_result': {
            #   '<metric_0>': {
            #       <key_0>: [<num_invalid>, <num_undefined>, ...],
            #       <key_1>: [<num_invalid>, <num_undefined>, ...],
            #   ...},
            # ...}
            'metric_to_result': h.dodol_from_lod(
                lod, metrics, 'result', (TEST_RESULT_MIN, TEST_RESULT_MAX)
            ),
            # Structure of 'result_to_metric': {
            #   '<metric_0>': [
            #       {<key_0>: <num_invalid>, <key_1>: <num_invalid>, ...},
            #       {<key_0>: <num_undefined>, <key_1>: <num_undefined>, ...},
            #   ...],
            # ...}
            'result_to_metric': h.dolod_from_lod(
                lod, metrics, 'result', (TEST_RESULT_MIN, TEST_RESULT_MAX)
            ),
        }

        if filepath:
            with open(Path(filepath), 'w') as f:
                json.dump(summary, f, indent=4)

        return summary


    @classmethod
    def from_summary(cls, summary):
        """Reconstruct an :class:`Evaluation` from a serialized summary.

        The summary can either be provided as a path to a JSON file or as a
        JSON string. The expected format is the dictionary produced by
        :meth:`result_summary`.

        Parameters
        ----------
        summary : str or pathlib.Path
            Either a path to a JSON file containing the serialized summary, or
            a JSON string encoding the summary dictionary.

        Returns
        -------
        Evaluation
            The reconstructed evaluation instance.

        Raises
        ------
        json.JSONDecodeError
            If ``summary`` is not a valid JSON string.
        KeyError
            If required keys are missing from the summary.

        Notes
        -----
        The returned evaluation contains the stored metadata and a population
        of :class:`TestCase` objects reconstructed via :meth:`TestCase.from_dict`.
        No diagrams are (re)computed.
        """

        def int_keys_hook(dictionary):
            return {
                int(key) if key.isdigit() else key: val
                for key, val in dictionary.items()
            }

        path = Path(summary)
        if path.is_file():
            with open(path, 'r') as fp:
                dictionary = json.load(fp, object_hook=int_keys_hook)
        else:
            dictionary = json.loads(summary, object_hook=int_keys_hook)

        evaluation = cls()
        evaluation.add_metadata(**dictionary['metadata'])
        evaluation.add_test_cases(
            [TestCase.from_dict(tc_dict) for tc_dict in dictionary['test_cases']]
        )
        return evaluation


    def filter_test_cases(self, *,
        test_result=None,
        runtime=(None, None),
        **metric_bounds
    ):
        """Filter the current population by result and metric bounds.

        Parameters
        ----------
        test_result : TestResult or list[TestResult] or None, optional
            If given, only include test cases whose :attr:`~TestCase.result` is
            in the provided value(s).
        runtime : tuple[int or None, int or None] or dict[int, tuple[int or None, int or None]], optional
            If a tuple, only include test cases whose runtime of the last stage is
            within the given (lower, upper) bounds. If a dict, it is interpreted
            as a mapping from stage index to (lower, upper) bounds on the runtime
            of that respective stage.
        **metric_bounds
            Additional metric bounds specified as keyword arguments
            ``metric=(lower, upper)``.

            Supported metric names are:

            - ``'num_vars'``
            - ``'num_gates'``
            - ``'num_not_gates'``
            - ``'num_and_gates'``
            - ``'num_or_gates'``
            - ``'num_xor_gates'``
            - ``'expr_depth'``
            - ``'num_literals'``

            Bounds are interpreted inclusively.

        Returns
        -------
        list[tuple[int, TestCase]]
            List of ``(index, test_case)`` pairs for all test cases that satisfy
            all provided constraints. The index refers to the 0-based position
            in :attr:`population`.

        Raises
        ------
        TypeError
            If a metric name is unknown.
        TypeError
            If any bounds value is not a tuple/list of length 2.
        """

        get = {
            'runtime': lambda tc: tc.timing['runtimes'] \
                .get(max(tc.timing['runtimes'].keys(), default=0), 0),
            'num_vars': lambda tc: tc.summary()['num_vars'],
            'num_gates': lambda tc: tc.summary()['num_gates'],
            'num_not_gates': lambda tc: tc.summary()['num_not_gates'],
            'num_and_gates': lambda tc: tc.summary()['num_and_gates'],
            'num_or_gates': lambda tc: tc.summary()['num_or_gates'],
            'num_xor_gates': lambda tc: tc.summary()['num_xor_gates'],
            'expr_depth': lambda tc: tc.summary()['expr_depth'],
            'num_literals': lambda tc: tc.summary()['num_literals'],
        }

        def filter_func(i_tc_pair):
            i, tc = i_tc_pair
            if test_result and tc.result not in test_result:
                return False
            if isinstance(runtime, dict):
                runtimes = tc.timing['runtimes']
                for i in runtime:
                    if i in runtimes:
                        value, (lower, upper) = runtimes[i], runtime[i]
                        if (lower is not None and value < lower) \
                            or (upper is not None and value > upper):
                            return False
            else:
                metric_bounds['runtime'] = runtime
            for metric, bounds in metric_bounds.items():
                if metric not in get:
                    raise TypeError(f'{metric} in an invalid metric name')
                if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                    raise TypeError(f'Bounds for {metric} should be a tuple/list of length 2')
                value, (lower, upper) = get[metric](tc), bounds
                if (lower is not None and value < lower) or (upper is not None and value > upper):
                    return False
            return True

        if test_result and isinstance(test_result, TestResult):
            test_result = [test_result]
        return list(filter(filter_func, enumerate(self.population)))


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
        self._result, self._timing = TestResult.UNDEFINED, {'runtimes': dict(), 'timeout': False}


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

    @property
    def timing(self):
        """dict: Timing metadata collected during :meth:`run`.

        Notes
        -----
        The dictionary has the following keys:

        ``'runtimes'``
            Mapping ``step_index -> runtime_ms`` for each executed pipeline
            step. Step indices are 1-based and correspond to the order in the
            evaluation pipeline.

        ``'timeout'``
            Boolean flag indicating whether any pipeline step hit the timeout
            limit.
        """

        return self._timing


    def __repr__(self):
        return f'TestCase({self._expr_string})'

    def __str__(self):
        return self.__repr__()

    def to_dict(self):
        """Serialize the test case to a dictionary.

        Returns
        -------
        dict
            Dictionary with the keys:

            ``'expr_string'``
                Canonical expression string.

            ``'result'``
                Name of the :class:`TestResult` value.

            ``'timing'``
                Timing metadata as returned by the :attr:`timing` property.
        """
        return {
            'expr_string': self._expr_string,
            'result': self._result.name,
            'timing': self._timing,
        }

    @classmethod
    def from_dict(cls, dictionary):
        """Construct a :class:`TestCase` from a serialized representation.

        Parameters
        ----------
        dictionary : dict
            Dictionary as produced by :meth:`to_dict`.

        Returns
        -------
        TestCase
            The reconstructed test case.

        Raises
        ------
        KeyError
            If required keys are missing.
        KeyError
            If ``dictionary['result']`` is not a valid :class:`TestResult` name.

        Notes
        -----
        Although the returned instance has set result and timing information,
        the ZH-diagram and subsequent representations are not created.
        """

        tc = cls(dictionary['expr_string'])
        tc._result = TestResult[dictionary['result']]
        tc._timing = dictionary['timing']
        return tc


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


    def run(self, verbosity=0, step_timeout_limit_ms=DEFAULT_STEP_TIMEOUT_LIMIT_MS):
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
            - FAIL1/FAIL2/FAIL3: assertion fails after step 1/2/3 respectively,
            or the step times out.

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
        self._timing['timeout'] = False
        for i, (act, assert_) in enumerate(pipeline):
            h.printv(f'\tRunning step {i+1}: {act.__name__}', verbosity=verbosity, level=2)
            t0 = perf_counter()
            try:
                with Timeout(step_timeout_limit_ms / 1000):
                    act()
            except TimeoutError:
                h.printv(f'\tTimeout after {step_timeout_limit_ms} milliseconds during {act.__name__}',
                    verbosity=verbosity, level=2)
                self._result = TestResult(i + 1)
                self._timing['timeout'] = True
                break
            except Exception as e:
                h.printv(f'[Test case {self._expr_string}] Error during {act.__name__}:\n{e}',
                    verbosity=verbosity, level=0)
                self._result = TestResult.INVALID
                break
            finally:
                self._timing['runtimes'][i+1] = \
                    min(int((perf_counter() - t0) * 1000), step_timeout_limit_ms)

            try:
                h.printv('\tAsserting result', verbosity=verbosity, level=2)
                assert_()
            except AssertionError:
                h.printv(f'\tAssertion failed after step {i+1}', verbosity=verbosity, level=2)
                # Determine TestResult enum value based on which step failed
                self._result = TestResult(i + 1)
                break

        h.printv(f'Test result: {self._result.name}\n', verbosity=verbosity, level=1)
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


    def draw(self, filepath=None):
        """Render a visualization of the test case and computed representations.

        The figure contains the matrix of the original boolean expression and,
        if available, the ZH-diagram, the simplified diagram, and the extracted
        circuit along with their associated matrices.

        Parameters
        ----------
        filepath : str or pathlib.Path or None, optional
            If given, save the rendered figure to this path using
            :meth:`matplotlib.figure.Figure.savefig`.

        Returns
        -------
        matplotlib.figure.Figure
            The created Matplotlib figure.

        Notes
        -----
        The diagram drawings are produced via :func:`pyzx.drawing.draw_matplotlib`.
        Matrix rendering uses LaTeX (``usetex=True``) and requires a working
        LaTeX installation in the runtime environment.
        """

        def obj_to_fig(obj):
            if isinstance(obj, BaseGraph):
                rows = [obj.row(v) for v in obj.vertex_set()]
                qubits = [obj.qubit(v) for v in obj.vertex_set()]
                width = max(rows) - min(rows) + 1
                height = max(qubits) - min(qubits) + 1
                return draw_matplotlib(
                    obj, h_edge_draw='box', show_scalar=True, labels=True, figsize=(width, height)
                )
            else:
                raise NotImplementedError(f'Drawing not implemented for type {type(obj)}')

        def matrix_to_latex(matrix):
            # use array environment because pmatrix does not allow more than 10 columns
            num_cols = 2 ** len(self._expr_sympy.vars)
            return zx_matrix_to_latex(matrix) \
                .replace('{equation}', '{equation*}') \
                .replace(r'\begin{pmatrix}',
                    r'\left( \begin{array}{@{}*{' + str(num_cols) + '}{c}@{}}') \
                .replace(r'\end{pmatrix}', r'\end{array} \right)') \
                .replace('\n', ' ') \
                if matrix is not None else 'N/A'

        def setup_axes(ax, title):
            ax.margins(0)
            ax.set_axis_off()
            ax.set_title(title, y=1.0, pad=-14)

        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        objs_matrices_titles = [
            (self._zh, self._zh_matrix, 'ZH-diagram:'),
            (self._simplified, self._simplified_matrix, 'Simplified diagram:'),
            (self._circuit, self._circuit_matrix, 'Extracted circuit:'),
        ]

        figs_latex = [
            (obj_to_fig(obj), matrix_to_latex(matrix), title) 
            for obj, matrix, title in objs_matrices_titles if obj
        ]
        figwidths = [fig.get_figwidth() for fig, _, _ in figs_latex]
        figheights = [fig.get_figheight() for fig, _, _ in figs_latex]
        fig = plt.figure(figsize=(
            max(figwidths)*0.625, sum(figheights) / 2 + 1
        ))
        spec = fig.add_gridspec(
            len(figs_latex) + 1, 2,
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0,
            width_ratios=[4, 1], height_ratios=[2] + figheights
        )

        ax0 = fig.add_subplot(spec[0, 0:2])
        expr_string = '$ ' + self._expr_string.replace('~', r'\neg ').replace('&', r'\wedge ') \
            .replace('|', r'\vee ').replace('^', r'\oplus ') + ' $'
        setup_axes(ax0, 'Test case ' + expr_string)
        ax0.text(0, 0, matrix_to_latex(self.expr_matrix),
            usetex=True, size='large', ha='left', va='center')

        for i, (fig_i, latex_i, title) in enumerate(figs_latex):
            ax_fig = fig.add_subplot(spec[i+1, 0])
            setup_axes(ax_fig, title)
            canvas = FigureCanvasAgg(fig_i)
            canvas.draw()
            ax_fig.imshow(np.asarray(canvas.buffer_rgba()))

            ax_latex = fig.add_subplot(spec[i+1, 1])
            setup_axes(ax_latex, 'Associated matrix:')
            ax_latex.text(0, 0, latex_i, usetex=True, size='large', ha='left', va='center')

        if filepath:
            fig.savefig(filepath, bbox_inches='tight', pad_inches=0)

        return fig


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
            ``timeout``
                Timeout value for the test case.
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
            ``num_literals``
                Number of literal occurrences in the expression.
        """

        gate_counts = self._expr_sympy.gate_counts()
        return {
            'expr_string': self._expr_string,
            'result': self._result.value,
            'timeout': self._timing['timeout'],
            'num_vars': len(self._expr_sympy.vars),
            'num_gates': sum(gate_counts.values()),
            'num_not_gates': gate_counts.get('Not', 0),
            'num_and_gates': gate_counts.get('And', 0),
            'num_or_gates': gate_counts.get('Or', 0),
            'num_xor_gates': gate_counts.get('Xor', 0),
            'expr_depth': self._expr_sympy.depth(),
            'num_literals': self._expr_sympy.num_literals(),
        }
