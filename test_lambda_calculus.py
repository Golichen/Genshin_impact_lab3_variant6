import unittest
import logging
from lambda_calculus import (
    LambdaTerm, Variable, Abstraction, Application,
    LambdaInterpreter, ReductionStrategy,
    parse_lambda_string,
    logger as lambda_logger
)
import sys
sys.setrecursionlimit(100000)

lambda_logger.setLevel(logging.DEBUG)  # Uncomment for verbose debug output


class TestLambdaCalculus(unittest.TestCase):

    def setUp(self):
        pass

    def assertLambdaEqual(
        self,
        term1: LambdaTerm,
        term2: LambdaTerm,
        msg=None
    ):
        self.assertEqual(term1, term2, msg)

    # --- 测试基本类 ---
    def test_variable(self):
        x = Variable("x")
        self.assertEqual("x", x.to_latex())
        self.assertEqual("Variable('x')", repr(x))
        # Line 30: E501 Split regex and message for assertRaisesRegex
        err_msg_empty = (
            r"Argument 'name' \('.*'\) of __init__ cannot be an empty "
            r"or whitespace-only string\."
        )
        with self.assertRaisesRegex(ValueError, err_msg_empty):
            Variable(name="")
        # Line 33: E501
        with self.assertRaisesRegex(ValueError, err_msg_empty):
            Variable(name="  ")

    def test_abstraction(self):
        x = Variable("x")
        term = Abstraction(x, x)
        self.assertEqual("\\lambda x.x", term.to_latex())
        # Line 40: E501
        repr_str = "Abstraction(Variable('x'), Variable('x'))"
        self.assertEqual(repr_str, repr(term))
        # Line 41: E501
        err_msg_var = (
            r"Keyword argument 'var' of __init__ expected .*, "
            r"got <class 'str'>"
        )
        with self.assertRaisesRegex(TypeError, err_msg_var):
            Abstraction(var="not_a_variable", body=x)  # type: ignore
        # Line 43: E501
        err_msg_body = (
            r"Keyword argument 'body' of __init__ expected .*, "
            r"got <class 'str'>"
        )
        with self.assertRaisesRegex(TypeError, err_msg_body):
            Abstraction(var=x, body="not_a_term")  # type: ignore

    def test_application(self):
        x = Variable("x")
        y = Variable("y")
        abs_term = Abstraction(x, x)
        app_term = Application(abs_term, y)
        self.assertEqual("(\\lambda x.x) y", app_term.to_latex())
        # Line 52: E501
        repr_str_app = (
            "Application(Abstraction(Variable('x'), Variable('x')), "
            "Variable('y'))"
        )
        self.assertEqual(repr_str_app, repr(app_term))
        # Line 53: E501
        err_msg_func = (
            r"Keyword argument 'func' of __init__ expected .*, "
            r"got <class 'str'>"
        )
        with self.assertRaisesRegex(TypeError, err_msg_func):
            Application(func="not_a_term", arg=y)  # type: ignore
        # Line 55: E501
        err_msg_arg = (
            r"Keyword argument 'arg' of __init__ expected .*, "
            r"got <class 'str'>"
        )
        with self.assertRaisesRegex(TypeError, err_msg_arg):
            Application(func=abs_term, arg="not_a_term")  # type: ignore

    # --- 测试解析器 ---
    def test_parse_simple_variable(self):
        term = parse_lambda_string("x")
        self.assertLambdaEqual(Variable("x"), term)

    def test_parse_simple_abstraction(self):
        term = parse_lambda_string("λx.x")
        expected = Abstraction(Variable("x"), Variable("x"))
        self.assertLambdaEqual(expected, term)

    def test_parse_simple_application(self):
        term = parse_lambda_string("x y")
        expected = Application(Variable("x"), Variable("y"))
        self.assertLambdaEqual(expected, term)

    def test_parse_left_associative_application(self):
        term = parse_lambda_string("x y z")
        # Line 75: E501
        expected = Application(
            Application(Variable("x"), Variable("y")), Variable("z")
        )
        self.assertLambdaEqual(expected, term)

    def test_parse_abstraction_with_application_body(self):
        term = parse_lambda_string("λx.y z")
        expected = Abstraction(
            Variable("x"), Application(Variable("y"), Variable("z"))
        )
        self.assertLambdaEqual(expected, term)

    def test_parse_application_with_abstraction_func(self):
        term = parse_lambda_string("(λx.x) y")
        expected = Application(
            Abstraction(Variable("x"), Variable("x")), Variable("y")
        )
        self.assertLambdaEqual(expected, term)

    def test_parse_application_with_abstraction_arg(self):
        term = parse_lambda_string("y (λx.x)")
        expected = Application(
            Variable("y"), Abstraction(Variable("x"), Variable("x"))
        )
        self.assertLambdaEqual(expected, term)

    def test_parse_nested_abstraction(self):
        term = parse_lambda_string("λx.λy.x y")
        expected = Abstraction(
            Variable("x"),
            Abstraction(
                Variable("y"),
                Application(Variable("x"), Variable("y"))
            )
        )
        self.assertLambdaEqual(expected, term)

    def test_parse_multi_var_abstraction(self):
        term = parse_lambda_string("λx y.x")
        # Line 101: E501
        expected = Abstraction(
            Variable("x"), Abstraction(Variable("y"), Variable("x"))
        )
        self.assertLambdaEqual(expected, term)

    def test_parse_complex_nested_expression(self):
        term = parse_lambda_string("(λx.λy.x y) z")
        # Line 107: E501
        expected = Application(
            Abstraction(
                Variable("x"),
                Abstraction(
                    Variable("y"),
                    Application(Variable("x"), Variable("y"))
                )
            ),
            Variable("z")
        )
        self.assertLambdaEqual(expected, term)

    def test_parse_s_combinator(self):
        term = parse_lambda_string("λx.λy.λz.x z (y z)")
        # Line 118 & 119: E501
        expected = Abstraction(
            Variable("x"),
            Abstraction(
                Variable("y"),
                Abstraction(
                    Variable("z"),
                    Application(
                        Application(Variable("x"), Variable("z")),
                        Application(Variable("y"), Variable("z"))
                    )
                )
            )
        )
        self.assertLambdaEqual(expected, term)

    # --- 测试解释器核心逻辑 ---
    def test_free_variables(self):
        interpreter = LambdaInterpreter()
        x, y, z = Variable("x"), Variable("y"), Variable("z")
        self.assertEqual(interpreter.free_variables(x), {x})
        term_abs = Abstraction(x, Application(x, y))
        self.assertEqual(interpreter.free_variables(term_abs), {y})
        term_app = Application(Abstraction(x, x), z)
        self.assertEqual(interpreter.free_variables(term_app), {z})

    def test_alpha_conversion(self):
        interpreter = LambdaInterpreter()
        x, y, z_new = Variable("x"), Variable("y"), Variable("z_new")
        original_term = Abstraction(x, Application(x, y))
        interpreter._collect_used_vars(original_term)
        interpreter.used_var_names.add(y.name)

        converted = interpreter.alpha_convert(original_term, z_new)
        expected = Abstraction(z_new, Application(z_new, y))
        self.assertLambdaEqual(expected, converted)
        # Line 143: E501
        step_msg = (
            f"{original_term.to_latex()} \\xrightarrow{{\\alpha}} "
            f"{expected.to_latex()}"
        )
        self.assertIn(step_msg, interpreter.steps)

    def test_substitute_no_capture(self):
        interpreter = LambdaInterpreter()
        x, y, z = Variable("x"), Variable("y"), Variable("z")
        term = Abstraction(y, Application(x, y))
        interpreter._collect_used_vars(term)
        interpreter._collect_used_vars(z)

        substituted = interpreter.substitute(term, x, z)
        expected = Abstraction(y, Application(z, y))
        self.assertLambdaEqual(expected, substituted)

    def test_substitute_capture_avoidance(self):
        interpreter = LambdaInterpreter()
        y_bound = Variable("y")
        x_outer = Variable("x")
        y_free_replacement = Variable("y")

        # Line 162: E501
        term_inner = Abstraction(y_bound, Application(x_outer, y_bound))

        interpreter._collect_used_vars(term_inner)
        interpreter._collect_used_vars(y_free_replacement)

        # Line 164 & 165: E501
        substituted_body_capture = interpreter.substitute(
            term_inner, x_outer, y_free_replacement
        )

        self.assertIsInstance(substituted_body_capture, Abstraction)
        new_binder_capture = substituted_body_capture.var
        # Line 169: E501
        self.assertNotEqual(
            new_binder_capture.name, y_bound.name,
            "Binder should have been renamed due to capture"
        )
        self.assertTrue(
            new_binder_capture.name.startswith(y_bound.name),
            "Renamed binder should start with original name"
        )
        # Line 173: E501
        expected_inner_app_capture = Application(
            y_free_replacement, new_binder_capture
        )
        self.assertLambdaEqual(
            substituted_body_capture.body, expected_inner_app_capture
        )

        interpreter_no_capture = LambdaInterpreter()
        y_bound_no_capture = Variable("y_bound")
        x_target_no_capture = Variable("x_target")
        # Line 177 & 178: E501
        y_different_replacement = Variable("y_replacement")

        term_no_capture = Abstraction(y_bound_no_capture, x_target_no_capture)

        interpreter_no_capture._collect_used_vars(term_no_capture)
        interpreter_no_capture._collect_used_vars(y_different_replacement)
        # Line 181 & 182: E501
        substituted_no_capture = interpreter_no_capture.substitute(
            term_no_capture, x_target_no_capture, y_different_replacement
        )

        self.assertIsInstance(substituted_no_capture, Abstraction)
        # Line 185: E501
        self.assertEqual(
            substituted_no_capture.var.name, y_bound_no_capture.name,
            "Binder should NOT be renamed if no capture"
        )
        self.assertLambdaEqual(
            substituted_no_capture.body, y_different_replacement
        )

    def test_beta_reduction_simple(self):
        interpreter = LambdaInterpreter()
        term = parse_lambda_string("(λx.x) y")
        interpreter._collect_used_vars(term)

        reduced = interpreter.beta_reduce_once(term)
        self.assertIsNotNone(reduced)
        self.assertLambdaEqual(Variable("y"), reduced)
        # Line 192 & 193: E501
        beta_step_msg = (
            f"{term.to_latex()} \\xrightarrow{{\\beta}} "
            f"{Variable('y').to_latex()}"
        )
        self.assertIn(beta_step_msg, interpreter.steps)

    def test_eta_conversion(self):
        interpreter = LambdaInterpreter()
        term_eta = parse_lambda_string("λx.(f x)")
        interpreter._collect_used_vars(term_eta)

        converted = interpreter.eta_convert_once(term_eta)
        self.assertIsNotNone(converted)
        self.assertLambdaEqual(Variable("f"), converted)
        # Line 197 & 199: E501
        eta_step_msg = (
            f"{term_eta.to_latex()} \\xrightarrow{{\\eta}} "
            f"{Variable('f').to_latex()}"
        )
        self.assertIn(eta_step_msg, interpreter.steps)

        term_no_eta = parse_lambda_string("λx.(x x)")
        interpreter._collect_used_vars(term_no_eta)
        self.assertIsNone(interpreter.eta_convert_once(term_no_eta))

        # Line 209: E501
        term_no_eta2 = parse_lambda_string("λx.((λy.x) x)")
        interpreter._collect_used_vars(term_no_eta2)
        self.assertIsNone(interpreter.eta_convert_once(term_no_eta2))

    # --- 测试解释器归约 ---
    def test_interpret_identity_app(self):
        interpreter = LambdaInterpreter(max_steps=10)
        term = parse_lambda_string("(λx.x) y")
        result = interpreter.interpret(term)
        self.assertLambdaEqual(Variable("y"), result)
        # Line 219: E501
        self.assertTrue(
            interpreter.steps[-1].strip().endswith("(Final Result)"),
            f"Last step was: {interpreter.steps[-1]}"
        )

    def test_interpret_normal_order_omega(self):
        interpreter = LambdaInterpreter(
            strategy=ReductionStrategy.NORMAL_ORDER, max_steps=5
        )
        term = parse_lambda_string("(λx.x x) (λx.x x)")
        result = interpreter.interpret(term)
        # Line 235: E501
        self.assertTrue(
            "(max steps reached)" in interpreter.steps[-1],
            f"Last step was: {interpreter.steps[-1]}"
        )
        self.assertIsInstance(result, Application)

    def test_interpret_applicative_order_omega(self):
        interpreter = LambdaInterpreter(
            strategy=ReductionStrategy.APPLICATIVE_ORDER, max_steps=5
        )
        term = parse_lambda_string("(λx.x x) (λx.x x)")
        result = interpreter.interpret(term)
        self.assertTrue(
            "(max steps reached)" in interpreter.steps[-1],
            f"Last step was: {interpreter.steps[-1]}"
        )

    def test_interpret_s_k_k_to_i(self):
        s_str = "λs_x.λs_y.λs_z.s_x s_z (s_y s_z)"
        k_str = "λk_a.λk_b.k_a"
        term_str = f"( ({s_str}) ({k_str}) ) ({k_str})"
        term = parse_lambda_string(term_str)
        interpreter = LambdaInterpreter(max_steps=50)
        result = interpreter.interpret(term)
        self.assertIsInstance(result, Abstraction)
        final_abs = result
        # Line 263: E501
        self.assertIsInstance(final_abs.body, Variable)  # type: ignore
        # Line 266 & 267: E501
        self.assertEqual(
            final_abs.var.name, final_abs.body.name  # type: ignore
        )

    def test_y_combinator(self):
        # Line 275: E501
        y_comb_str = "λy_f.(λy_x.y_f (y_x y_x)) (λy_x.y_f (y_x y_x))"
        Y_COMBINATOR = parse_lambda_string(y_comb_str)
        # Line 278: E501
        const_func_gen_str = "λrec.λval.val"
        CONST_FUNC_GEN = parse_lambda_string(const_func_gen_str)
        Y_CONST = Application(Y_COMBINATOR, CONST_FUNC_GEN)

        interpreter_yconst = LambdaInterpreter(max_steps=60)
        result_Y_CONST = interpreter_yconst.interpret(Y_CONST)
        # Line 280: E501
        expected_type_msg = (
            f"Expected Abstraction, got {type(result_Y_CONST)}: "
            f"{result_Y_CONST.to_latex()}"
        )
        self.assertIsInstance(
            result_Y_CONST,
            Abstraction,
            expected_type_msg
        )
        final_abs_const = result_Y_CONST
        expected_body_type_msg = (
            f"Expected body to be Variable, "
            f"got {type(final_abs_const.body)}"
        )
        self.assertIsInstance(
            final_abs_const.body, Variable,
            expected_body_type_msg
        )
        # Line 293: E501
        self.assertEqual(
            final_abs_const.var.name, final_abs_const.body.name,
            "Expected var and body name to match in Id function"
        )

    # Line 381: E301 Ensure 1 blank line before new section
    # --- 测试装饰器 ---
    def test_input_validation_decorators_type_check(self):
        err_msg_strat = (
            r"Keyword argument 'strategy' of __init__ expected "
            r"<enum 'ReductionStrategy'>, got <class 'str'>"
        )
        with self.assertRaisesRegex(TypeError, err_msg_strat):
            LambdaInterpreter(strategy="not_a_strategy")

        err_msg_steps = (
            r"Keyword argument 'max_steps' of __init__ expected "
            r"<class 'int'>, got <class 'str'>"
        )
        with self.assertRaisesRegex(TypeError, err_msg_steps):
            LambdaInterpreter(max_steps="not_an_int")

        err_msg_s_type = (
            r"Keyword argument 's' of parse_lambda_string expected "
            r"<class 'str'>, got <class 'int'>"
        )
        with self.assertRaisesRegex(TypeError, err_msg_s_type):
            parse_lambda_string(s=123)

    def test_input_validation_decorators_non_empty_string(self):
        # Line 400: E501
        err_msg_s_empty_pls = (
            r"Argument 's' \('.*'\) of parse_lambda_string cannot be an "
            r"empty or whitespace-only string\."
        )
        with self.assertRaisesRegex(ValueError, err_msg_s_empty_pls):
            parse_lambda_string(s="")
        # Line 403: E501
        with self.assertRaisesRegex(ValueError, err_msg_s_empty_pls):
            parse_lambda_string(s="   ")

        # Line 408: E501
        err_msg_name_empty_var = (
            r"Argument 'name' \('.*'\) of __init__ cannot be an empty "
            r"or whitespace-only string\."
        )
        with self.assertRaisesRegex(ValueError, err_msg_name_empty_var):
            Variable(name=" ")

    # --- 测试可视化步骤 ---
    def test_visualization_steps_recorded(self):
        interpreter = LambdaInterpreter(max_steps=10)
        term = parse_lambda_string("(λx.x) y")
        result = interpreter.interpret(term)

        self.assertTrue(len(interpreter.steps) >= 2)
        self.assertTrue("(Initial Term)" in interpreter.steps[0])
        self.assertTrue("(Final Result)" in interpreter.steps[-1].strip())
        self.assertTrue("\\xrightarrow{\\beta}" in interpreter.steps[1])


if __name__ == '__main__':
    unittest.main()
