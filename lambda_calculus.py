import logging
import re
from enum import Enum
from typing import Set, Union, List, Tuple, Optional
import functools  # 用于装饰器


# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,  # logging.DEBUG for more verbose output
    format=(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
)
logger = logging.getLogger("LambdaCalculus")


# --- 装饰器 ---
def type_check(*p_arg_types, **kw_arg_types):
    # distinguish from user-facing names
    """
    装饰器：检查函数/方法参数的类型。
    第一个位置参数 (self/cls) 会被跳过，如果 p_arg_types 对应的是用户参数。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查位置参数 (args[0] is self/cls, so user args start from args[1])
            # p_arg_types corresponds to types for args[1], args[2], ...
            for i, (arg_val, expected_type) in enumerate(
                zip(args[1:], p_arg_types)
            ):
                if not isinstance(arg_val, expected_type):
                    expected_repr = repr(expected_type)
                    actual_repr = repr(type(arg_val))
                    msg = (
                        f"Positional argument {i + 1} of {func.__name__} "
                        f"expected {expected_repr}, got {actual_repr}"
                    )
                    raise TypeError(msg)

            # 检查关键字参数
            for name, expected_type in kw_arg_types.items():
                if name in kwargs and \
                   not isinstance(kwargs[name], expected_type):
                    expected_repr = repr(expected_type)
                    actual_repr = repr(type(kwargs[name]))
                    msg = (
                        f"Keyword argument '{name}' of {func.__name__} "
                        f"expected {expected_repr}, got {actual_repr}"
                    )
                    raise TypeError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def non_empty_string_check(arg_name: str):
    # Simplified to only check by name for clarity
    """
    装饰器：检查指定的命名字符串参数是否非空。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            value_to_check = None
            param_description = f"Argument '{arg_name}'"

            if arg_name in kwargs:
                value_to_check = kwargs[arg_name]
            else:
                pass

            # find 'arg_name' in kwargs or in positional
            import inspect
            sig = inspect.signature(func)
            # Use bind_partial to avoid error on unbound
            bound_args = sig.bind_partial(*args, **kwargs)

            if arg_name in bound_args.arguments:
                value_to_check = bound_args.arguments[arg_name]

            if value_to_check is not None:
                if not isinstance(value_to_check, str):
                    msg = (
                        f"{param_description} of {func.__name__} must be "
                        f"a string."
                    )
                    raise TypeError(msg)
                if not value_to_check.strip():
                    value_repr = repr(value_to_check)
                    msg = (
                        f"{param_description} ({value_repr}) of "
                        f"{func.__name__} cannot be an empty or "
                        f"whitespace-only string."
                    )
                    raise ValueError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# --- Lambda 项的定义 ---
class LambdaTerm:
    """Lambda 项的基类"""

    def to_latex(self) -> str:
        raise NotImplementedError("Subclasses must implement to_latex")

    def to_input_string(self) -> str:  # New method
        """将 Lambda 项转换为可供解析器解析的字符串"""
        raise NotImplementedError("Subclasses must implement to_input_string")

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self) -> str:
        raise NotImplementedError("Subclasses must implement __repr__")


class Variable(LambdaTerm):
    @type_check(name=str)  # Check 'name' if passed as keyword
    @non_empty_string_check(arg_name="name")
    def __init__(self, name: str):
        _name = name.strip()
        if not _name:
            raise ValueError(
                "Variable name cannot be empty or whitespace (internal check)."
            )

        if not re.match(r"^(?!^\d+$)[a-zA-Z_][a-zA-Z0-9_]*$", _name):
            parts = _name.split('_')
            is_complex_valid = False
            if len(parts) > 1 and parts[-1].isdigit():
                first_part = "_".join(parts[:-1])
                if re.match(r"^(?!^\d+$)[a-zA-Z_][a-zA-Z0-9_]*$", first_part):
                    is_complex_valid = True

            if not is_complex_valid:
                # Line 118 related fix
                logger.warning(
                    "Variable name '%s' has non-standard chars or format.",
                    _name
                )
        self.name = _name

    def to_latex(self) -> str:
        return self.name.replace("_", r"\_")

    def to_input_string(self) -> str:
        return self.name

    def __repr__(self):
        return f"Variable('{self.name}')"


class Abstraction(LambdaTerm):
    @type_check(var=Variable, body=LambdaTerm)
    def __init__(self, var: Variable, body: LambdaTerm):
        if not isinstance(var, Variable):
            raise TypeError(
                "Bound variable in Abstraction must be a Variable instance."
            )
        if not isinstance(body, LambdaTerm):
            raise TypeError(
                "Body of Abstraction must be a LambdaTerm instance."
            )
        self.var = var
        self.body = body

    def to_latex(self) -> str:
        current_term = self
        latex_parts = []
        while isinstance(current_term, Abstraction):
            latex_parts.append(f"\\lambda {current_term.var.to_latex()}.")
            current_term = current_term.body
        latex_parts.append(current_term.to_latex())
        return "".join(latex_parts)

    def to_input_string(self) -> str:
        current = self
        parts = []
        while isinstance(current, Abstraction):
            # Line 168 related: ensure parts.append(...) is not too long
            var_input_str = current.var.to_input_string()
            parts.append(f"λ{var_input_str}.")
            current = current.body
        parts.append(current.to_input_string())
        return "".join(parts).rstrip(".")

    def __repr__(self):
        return f"Abstraction({repr(self.var)}, {repr(self.body)})"


class Application(LambdaTerm):
    @type_check(func=LambdaTerm, arg=LambdaTerm)
    def __init__(self, func: LambdaTerm, arg: LambdaTerm):
        if not isinstance(func, LambdaTerm):
            raise TypeError(
                "Function part of Application must be a LambdaTerm instance."
            )
        if not isinstance(arg, LambdaTerm):
            raise TypeError(
                "Argument part of Application must be a LambdaTerm instance."
            )
        self.func = func
        self.arg = arg

    def to_latex(self) -> str:
        func_is_complex = isinstance(self.func, (Abstraction, Application))
        func_str = (
            f"({self.func.to_latex()})"
            if func_is_complex
            else self.func.to_latex()
        )
        arg_is_complex = isinstance(self.arg, (Abstraction, Application))
        arg_str = (
            f"({self.arg.to_latex()})"
            if arg_is_complex
            else self.arg.to_latex()
        )
        return f"{func_str} {arg_str}"

    def to_input_string(self) -> str:
        func_input = self.func.to_input_string()
        arg_input = self.arg.to_input_string()

        if isinstance(self.func, Abstraction):
            func_str = f"({func_input})"
        else:
            func_str = func_input

        if isinstance(self.arg, (Abstraction, Application)):
            arg_str = f"({arg_input})"
        else:
            arg_str = arg_input
        return f"{func_str} {arg_str}"

    def __repr__(self):
        return f"Application({repr(self.func)}, {repr(self.arg)})"


class ReductionStrategy(Enum):
    NORMAL_ORDER = 1
    APPLICATIVE_ORDER = 2


class LambdaInterpreter:
    @type_check(strategy=ReductionStrategy, max_steps=int)
    def __init__(
        self,
        strategy: ReductionStrategy = ReductionStrategy.NORMAL_ORDER,
        max_steps: int = 100
    ):
        self.strategy = strategy
        self.max_steps = max_steps
        self.steps: List[str] = []
        self.used_var_names: Set[str] = set()
        logger.info(
            "Interpreter initialized with strategy: %s, max_steps: %s",
            self.strategy, self.max_steps
        )

    def _collect_used_vars(self, term: LambdaTerm):
        if isinstance(term, Variable):
            self.used_var_names.add(term.name)
        elif isinstance(term, Abstraction):
            self.used_var_names.add(term.var.name)
            self._collect_used_vars(term.body)
        elif isinstance(term, Application):
            self._collect_used_vars(term.func)
            self._collect_used_vars(term.arg)

    def fresh_var(self, base_name: str = "v") -> Variable:
        i = 0
        new_name = base_name
        if base_name.isdigit():
            base_name = f"v_{base_name}"
            new_name = base_name

        while new_name in self.used_var_names:
            new_name = f"{base_name}_{i}"
            i += 1
        self.used_var_names.add(new_name)
        return Variable(new_name)

    def free_variables(self, term: LambdaTerm) -> Set[Variable]:
        if isinstance(term, Variable):
            return {term}
        elif isinstance(term, Abstraction):
            return self.free_variables(term.body) - {term.var}
        elif isinstance(term, Application):
            return self.free_variables(term.func) | \
                   self.free_variables(term.arg)
        return set()

    def substitute(
        self,
        term: LambdaTerm,
        target_var: Variable,
        replacement: LambdaTerm
    ) -> LambdaTerm:
        if isinstance(term, Variable):
            return replacement if term == target_var else term
        elif isinstance(term, Abstraction):
            if term.var == target_var:
                return term
            else:
                fv_replacement = self.free_variables(replacement)
                if term.var in fv_replacement:
                    term_latex = term.to_latex()
                    replacement_latex = replacement.to_latex()
                    log_msg_part1 = (
                        f"Alpha conversion needed for binder {term.var.name}"
                    )
                    log_msg_part2 = (
                        f" in {term_latex} due to replacement {replacement_latex}"
                    )
                    logger.debug(log_msg_part1 + log_msg_part2)

                    new_binder = self.fresh_var(term.var.name)
                    logger.debug(
                        "Generated fresh variable %s for %s",
                        new_binder.name, term.var.name
                    )
                    body_renamed = self.substitute(
                        term.body, term.var, new_binder
                    )
                    return Abstraction(
                        new_binder,
                        self.substitute(
                            body_renamed, target_var, replacement
                        )
                    )
                else:
                    return Abstraction(
                        term.var,
                        self.substitute(
                            term.body, target_var, replacement
                        )
                    )
        elif isinstance(term, Application):
            return Application(
                self.substitute(term.func, target_var, replacement),
                self.substitute(term.arg, target_var, replacement)
            )
        return term

    def alpha_convert(self, term: Abstraction, new_var: Variable) -> Abstraction:
        if not isinstance(term, Abstraction):
            raise TypeError(
                "Alpha conversion can only be applied to Abstraction."
            )
        logger.info(
            "Alpha converting: %s with new var %s",
            term.to_latex(), new_var.name
        )
        old_var = term.var
        new_body = self.substitute(term.body, old_var, new_var)
        converted_term = Abstraction(new_var, new_body)
        # Line 305 fix:
        term_latex = term.to_latex()
        converted_latex = converted_term.to_latex()
        step_str = f"{term_latex} \\xrightarrow{{\\alpha}} {converted_latex}"
        self.steps.append(step_str)
        return converted_term

    def beta_reduce_once(self, term: LambdaTerm) -> Optional[LambdaTerm]:
        if isinstance(term, Application) and isinstance(term.func, Abstraction):
            original_latex = term.to_latex()
            reduced_term = self.substitute(
                term.func.body, term.func.var, term.arg
            )
            reduced_latex = reduced_term.to_latex()
            logger.info(
                "Beta reducing: %s -> %s", original_latex, reduced_latex
            )
            # Line 332 fix:
            step_str = f"{original_latex} \\xrightarrow{{\\beta}} {reduced_latex}"
            self.steps.append(step_str)
            return reduced_term
        return None

    def eta_convert_once(self, term: LambdaTerm) -> Optional[LambdaTerm]:
        if isinstance(term, Abstraction):
            if isinstance(term.body, Application):
                if isinstance(term.body.arg, Variable) and \
                   term.body.arg == term.var:
                    m_term = term.body.func
                    if term.var not in self.free_variables(m_term):
                        original_latex = term.to_latex()
                        m_term_latex = m_term.to_latex()
                        logger.info(
                            "Eta converting: %s -> %s",
                            original_latex, m_term_latex
                        )
                        step_str = (
                            f"{original_latex} \\xrightarrow{{\\eta}} "
                            f"{m_term_latex}"
                        )
                        self.steps.append(step_str)
                        return m_term
        return None

    @type_check(term=LambdaTerm, current_step=int)
    def reduce(self, term: LambdaTerm, current_step: int = 0) -> LambdaTerm:
        if current_step >= self.max_steps:
            # Line 363 fix:
            term_ltx = term.to_latex()
            logger.warning(
                "Reduction limit reached at step %s for term: %s",
                current_step, term_ltx
            )
            max_steps_msg = term_ltx + " (max steps reached)"
            if not self.steps or \
               not self.steps[-1].strip().endswith("(max steps reached)"):
                self.steps.append(max_steps_msg)
            return term

        beta_reduced = self.beta_reduce_once(term)
        if beta_reduced is not None:
            return self.reduce(beta_reduced, current_step + 1)

        if self.strategy == ReductionStrategy.NORMAL_ORDER:
            if isinstance(term, Application):
                reduced_func = self.reduce(term.func, current_step + 1)
                if reduced_func != term.func:
                    # Line 382 related fix:
                    new_app = Application(reduced_func, term.arg)
                    return self.reduce(new_app, current_step + 1)

                reduced_arg = self.reduce(term.arg, current_step + 1)
                if reduced_arg != term.arg:
                    new_app = Application(term.func, reduced_arg)
                    return self.reduce(new_app, current_step + 1)

            elif isinstance(term, Abstraction):
                reduced_body = self.reduce(term.body, current_step + 1)
                if reduced_body != term.body:
                    return Abstraction(term.var, reduced_body)
                return Abstraction(term.var, reduced_body)

        elif self.strategy == ReductionStrategy.APPLICATIVE_ORDER:
            if isinstance(term, Application):
                original_func = term.func
                original_arg = term.arg

                reduced_func = self.reduce(term.func, current_step + 1)
                reduced_arg = self.reduce(term.arg, current_step + 1)

                current_app = Application(reduced_func, reduced_arg)
                beta_after_args = self.beta_reduce_once(current_app)
                if beta_after_args is not None:
                    return self.reduce(beta_after_args, current_step + 1)

                if reduced_func != original_func or \
                   reduced_arg != original_arg:
                    return current_app

            elif isinstance(term, Abstraction):
                reduced_body = self.reduce(term.body, current_step + 1)
                if reduced_body != term.body:
                    return Abstraction(term.var, reduced_body)

        eta_converted = self.eta_convert_once(term)
        if eta_converted is not None:
            return self.reduce(eta_converted, current_step + 1)

        logger.debug(
            "Term %s is stable at step %s", term.to_latex(), current_step
        )
        return term

    @type_check(term=LambdaTerm)
    def interpret(self, term: LambdaTerm) -> LambdaTerm:
        # Line 436 fix:
        logger.info(
            "Interpreting term: %s using %s strategy.",
            term.to_latex(), self.strategy
        )
        self.steps = [term.to_latex() + " (Initial Term)"]
        self.used_var_names = set()
        self._collect_used_vars(term)

        result = self.reduce(term, 0)
        result_latex = result.to_latex()

        final_step_str = result_latex + " (Final Result)"
        last_step_is_final_or_max = (
            self.steps and
            self.steps[-1].startswith(result_latex) and
            ("(Final Result)" in self.steps[-1] or
             "(max steps reached)" in self.steps[-1])
        )

        if not self.steps or not last_step_is_final_or_max:
            if not (self.steps and "(max steps reached)" in self.steps[-1]):
                self.steps.append(final_step_str)
        elif "(max steps reached)" not in self.steps[-1]:
            self.steps[-1] = final_step_str

        logger.info(
            "Interpretation finished. Result: %s", result_latex
        )
        return result


# --- 输入解析器 (Parser remains largely the same as provided previously) ---
# 令牌类型
LAMBDA = 'LAMBDA'  # λ or \
VAR = 'VAR'      # x, y, myVar
DOT = 'DOT'      # .
LPAREN = 'LPAREN'  # (
RPAREN = 'RPAREN'  # )
EOF = 'EOF'      # End of input


def tokenize(s: str) -> List[Tuple[str, str]]:
    logger.debug("Tokenizing string: '%s'", s)
    tokens: List[Tuple[str, str]] = []
    i = 0
    while i < len(s):
        if s.startswith("\\lambda", i):
            tokens.append((LAMBDA, "\\lambda"))
            i += len("\\lambda")
            continue

        char = s[i]

        if char.isspace():
            i += 1
            continue
        elif char == 'λ':
            tokens.append((LAMBDA, 'λ'))
            i += 1
        elif char == '.':
            tokens.append((DOT, char))
            i += 1
        elif char == '(':
            tokens.append((LPAREN, char))
            i += 1
        elif char == ')':
            tokens.append((RPAREN, char))
            i += 1
        # Line 492 fix: Broke long comment
        # Adjusted to ensure var names like 's_x' tokenized as one VAR
        elif char.isalpha() or char == '_':
            name = char
            i += 1
            # Old: while i < len(s) and (s[i].isalnum() or s[i] == '_'):
            while i < len(s) and (s[i].isalnum() or s[i] == '_'):
                name += s[i]
                i += 1
            tokens.append((VAR, name))
        else:
            # Line 500 fix: ensure SyntaxError msg construction is short
            msg = f"Unexpected character: {char} at position {i}"
            raise SyntaxError(msg)
    tokens.append((EOF, ""))
    logger.debug("Tokens: %s", tokens)
    return tokens


class Parser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0
        # Line 530 E501/E261 fix: Split conditional assignment
        if self.tokens:
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = (EOF, "")

    # Line 534 E303 fix: Ensure one blank line between methods
    def _advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = (EOF, "")

    def _eat(self, token_type: str):
        if self.current_token[0] == token_type:
            self._advance()
        else:
            msg = (
                f"Expected token {token_type} but "
                f"got {self.current_token[0]} ('{self.current_token[1]}') "
                f"at pos {self.pos}"
            )
            raise SyntaxError(msg)

    def _parse_atom(self) -> LambdaTerm:
        token_type, token_value = self.current_token
        if token_type == VAR:
            self._eat(VAR)
            return Variable(token_value)
        elif token_type == LPAREN:
            self._eat(LPAREN)
            term = self._parse_expression()
            self._eat(RPAREN)
            return term
        else:
            msg = (
                f"Unexpected token {token_type} ('{token_value}') "
                f"when expecting an atom."
            )
            raise SyntaxError(msg)

    def _parse_application(self) -> LambdaTerm:
        if self.current_token[0] == LAMBDA:
            term = self._parse_abstraction()
        else:
            term = self._parse_atom()

        while self.current_token[0] not in [RPAREN, DOT, EOF]:
            # Check for valid argument start
            is_var_or_lparen = (self.current_token[0] == VAR or
                                self.current_token[0] == LPAREN)
            is_lambda = self.current_token[0] == LAMBDA

            if is_var_or_lparen:
                arg = self._parse_atom()
                term = Application(term, arg)
            elif is_lambda:
                arg = self._parse_abstraction()
                term = Application(term, arg)
            else:
                break
        return term

    def _parse_abstraction(self) -> Abstraction:
        self._eat(LAMBDA)
        variables: List[Variable] = []
        while self.current_token[0] == VAR:
            variables.append(Variable(self.current_token[1]))
            self._eat(VAR)

        if not variables:
            raise SyntaxError("Expected at least one variable after lambda.")

        self._eat(DOT)
        body = self._parse_expression()

        # Build nested abstractions from right to left
        # e.g., λx y z.body -> Abstraction(x, Abstraction(y, Abstraction(z, body)))
        current_body = body
        for var in reversed(variables):
            current_body = Abstraction(var, current_body)
        return current_body  # type: ignore

    def _parse_expression(self) -> LambdaTerm:
        if self.current_token[0] == LAMBDA:
            return self._parse_abstraction()
        else:
            return self._parse_application()

    def parse(self) -> LambdaTerm:
        logger.info("Starting parsing process.")
        if not self.tokens or self.tokens[0][0] == EOF:
            raise SyntaxError("Cannot parse empty input.")
        term = self._parse_expression()
        if self.current_token[0] != EOF:
            remaining_tokens = self.tokens[self.pos:]
            logger.warning(
                "Input partially parsed. Extra tokens remain: %s",
                remaining_tokens
            )
        logger.info("Parsing successful. Result: %s", term.to_latex())
        return term


@type_check(s=str)
@non_empty_string_check(arg_name="s")
def parse_lambda_string(s: str) -> LambdaTerm:
    tokens = tokenize(s)
    parser = Parser(tokens)
    return parser.parse()


if __name__ == '__main__':
    print("Lambda Calculus Interpreter Examples")
    # Line 601 related: Y_COMBINATOR_STR is fine as is.
    # Parenthesized multi-line string literal.
    Y_COMBINATOR_STR = (
        "λy_f.(λy_x.y_f (y_x y_x)) "
        "(λy_x.y_f (y_x y_x))"
    )
    G_var_str = "G"
    Y_G_app_str = f"({Y_COMBINATOR_STR}) {G_var_str}"
    print(f"\nExample Y G: {Y_G_app_str}")

    Y_G_term = parse_lambda_string(Y_G_app_str)
    interpreter_yg = LambdaInterpreter(max_steps=10)
    result_yg = interpreter_yg.interpret(Y_G_term)
    print(f"Parsed Y G: {Y_G_term.to_latex()}")
    print(f"Result of Y G: {result_yg.to_latex()}")
    print("Steps (LaTeX) for Y G:")
    for step_latex in interpreter_yg.steps:
        print(f"  $ {step_latex} $")

    omega_str = "(λx.x x) (λx.x x)"
    print(f"\nOmega: {omega_str}")
    omega_term = parse_lambda_string(omega_str)
    interpreter_omega = LambdaInterpreter(
        strategy=ReductionStrategy.NORMAL_ORDER,
        max_steps=5
    )
    result_omega = interpreter_omega.interpret(omega_term)
    print(f"Result (Normal): {result_omega.to_latex()}")
    print("Steps (LaTeX):")
    for step_latex in interpreter_omega.steps:
        print(f"  $ {step_latex} $")
