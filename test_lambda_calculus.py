# test_lambda_calculus_fixed.py
import unittest
import logging
from lambda_calculus import (
    LambdaTerm, Variable, Abstraction, Application,
    LambdaInterpreter, ReductionStrategy,
    parse_lambda_string,
    # type_check, non_empty_string_check, # Removed if not directly used in this file
    logger as lambda_logger
)
import sys
sys.setrecursionlimit(100000)  # 提高递归深度限制

lambda_logger.setLevel(logging.DEBUG) # Uncomment for verbose debug output

class TestLambdaCalculusFixed(unittest.TestCase):

    def setUp(self):
        pass

    def assertLambdaEqual(self, term1: LambdaTerm, term2: LambdaTerm, msg=None):
        self.assertEqual(term1, term2, msg)

    # --- 测试基本类 ---
    def test_variable(self):
        x = Variable("x")
        self.assertEqual("x", x.to_latex())
        self.assertEqual("Variable('x')", repr(x))
        with self.assertRaisesRegex(ValueError,
                                    r"Argument 'name' \('.*'\) of __init__ cannot be an empty or whitespace-only string\."):
            Variable(name="")  # Pass as keyword to match decorator's expectation
        with self.assertRaisesRegex(ValueError,
                                    r"Argument 'name' \('.*'\) of __init__ cannot be an empty or whitespace-only string\."):
            Variable(name="  ")  # Pass as keyword

    def test_abstraction(self):
        x = Variable("x")
        term = Abstraction(x, x)
        self.assertEqual("\\lambda x.x", term.to_latex())
        self.assertEqual("Abstraction(Variable('x'), Variable('x'))", repr(term))
        with self.assertRaisesRegex(TypeError, r"Keyword argument 'var' of __init__ expected .*, got <class 'str'>"):
            Abstraction(var="not_a_variable", body=x)  # type: ignore
        with self.assertRaisesRegex(TypeError, r"Keyword argument 'body' of __init__ expected .*, got <class 'str'>"):
            Abstraction(var=x, body="not_a_term")  # type: ignore

    def test_application(self):
        x = Variable("x")
        y = Variable("y")
        abs_term = Abstraction(x, x)
        app_term = Application(abs_term, y)
        self.assertEqual("(\\lambda x.x) y", app_term.to_latex())
        self.assertEqual("Application(Abstraction(Variable('x'), Variable('x')), Variable('y'))", repr(app_term))
        with self.assertRaisesRegex(TypeError, r"Keyword argument 'func' of __init__ expected .*, got <class 'str'>"):
            Application(func="not_a_term", arg=y)  # type: ignore
        with self.assertRaisesRegex(TypeError, r"Keyword argument 'arg' of __init__ expected .*, got <class 'str'>"):
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
        expected = Application(Application(Variable("x"), Variable("y")), Variable("z"))
        self.assertLambdaEqual(expected, term)

    def test_parse_abstraction_with_application_body(self):
        term = parse_lambda_string("λx.y z")
        expected = Abstraction(Variable("x"), Application(Variable("y"), Variable("z")))
        self.assertLambdaEqual(expected, term)

    def test_parse_application_with_abstraction_func(self):
        term = parse_lambda_string("(λx.x) y")
        expected = Application(Abstraction(Variable("x"), Variable("x")), Variable("y"))
        self.assertLambdaEqual(expected, term)

    def test_parse_application_with_abstraction_arg(self):
        term = parse_lambda_string("y (λx.x)")
        expected = Application(Variable("y"), Abstraction(Variable("x"), Variable("x")))
        self.assertLambdaEqual(expected, term)

    def test_parse_nested_abstraction(self):
        term = parse_lambda_string("λx.λy.x y")
        expected = Abstraction(Variable("x"),
                               Abstraction(Variable("y"), Application(Variable("x"), Variable("y"))))
        self.assertLambdaEqual(expected, term)

    def test_parse_multi_var_abstraction(self):
        term = parse_lambda_string("λx y.x")
        expected = Abstraction(Variable("x"), Abstraction(Variable("y"), Variable("x")))
        self.assertLambdaEqual(expected, term)

    def test_parse_complex_nested_expression(self):
        term = parse_lambda_string("(λx.λy.x y) z")
        expected = Application(
            Abstraction(Variable("x"), Abstraction(Variable("y"), Application(Variable("x"), Variable("y")))),
            Variable("z")
        )
        self.assertLambdaEqual(expected, term)

    def test_parse_s_combinator(self):
        term = parse_lambda_string("λx.λy.λz.x z (y z)")
        expected = Abstraction(Variable("x"),
                               Abstraction(Variable("y"),
                                           Abstraction(Variable("z"),
                                                       Application(
                                                           Application(Variable("x"), Variable("z")),
                                                           Application(Variable("y"), Variable("z"))
                                                       ))))
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
        self.assertIn(f"{original_term.to_latex()} \\xrightarrow{{\\alpha}} {expected.to_latex()}", interpreter.steps)

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

        term_inner = Abstraction(y_bound, Application(x_outer, y_bound))  # λy.x y

        interpreter._collect_used_vars(term_inner)  # Collect vars from the term being substituted into
        interpreter._collect_used_vars(y_free_replacement)  # And from the replacement term

        # Case 1: Capture occurs, binder should be renamed
        # (λy.x y)[x := y]  => λy_0. y y_0
        substituted_body_capture = interpreter.substitute(term_inner, x_outer, y_free_replacement)

        self.assertIsInstance(substituted_body_capture, Abstraction)
        new_binder_capture = substituted_body_capture.var
        self.assertNotEqual(new_binder_capture.name, y_bound.name, "Binder should have been renamed due to capture")
        self.assertTrue(new_binder_capture.name.startswith(y_bound.name),
                        "Renamed binder should start with original name")

        expected_inner_app_capture = Application(y_free_replacement, new_binder_capture)
        self.assertLambdaEqual(substituted_body_capture.body, expected_inner_app_capture)

        # Case 2: No capture, binder should NOT be renamed
        # (λy_bound. x_target)[x_target := y_different_replacement] => λy_bound. y_different_replacement
        interpreter_no_capture = LambdaInterpreter()  # Fresh interpreter for clean used_vars
        y_bound_no_capture = Variable("y_bound")
        x_target_no_capture = Variable("x_target")
        y_different_replacement = Variable("y_replacement")  # Different name, won't capture y_bound

        term_no_capture = Abstraction(y_bound_no_capture, x_target_no_capture)

        interpreter_no_capture._collect_used_vars(term_no_capture)
        interpreter_no_capture._collect_used_vars(y_different_replacement)

        substituted_no_capture = interpreter_no_capture.substitute(term_no_capture, x_target_no_capture,
                                                                   y_different_replacement)

        self.assertIsInstance(substituted_no_capture, Abstraction)
        # ** CORRECTED ASSERTION **
        self.assertEqual(substituted_no_capture.var.name, y_bound_no_capture.name,
                         "Binder should NOT be renamed if no capture")
        self.assertLambdaEqual(substituted_no_capture.body, y_different_replacement)

    def test_beta_reduction_simple(self):
        interpreter = LambdaInterpreter()
        term = parse_lambda_string("(λx.x) y")
        interpreter._collect_used_vars(term)

        reduced = interpreter.beta_reduce_once(term)
        self.assertIsNotNone(reduced)
        self.assertLambdaEqual(Variable("y"), reduced)
        self.assertIn(f"{term.to_latex()} \\xrightarrow{{\\beta}} {Variable('y').to_latex()}", interpreter.steps)

    def test_eta_conversion(self):
        interpreter = LambdaInterpreter()
        term_eta = parse_lambda_string("λx.(f x)")
        interpreter._collect_used_vars(term_eta)

        converted = interpreter.eta_convert_once(term_eta)
        self.assertIsNotNone(converted)
        self.assertLambdaEqual(Variable("f"), converted)
        self.assertIn(f"{term_eta.to_latex()} \\xrightarrow{{\\eta}} {Variable('f').to_latex()}", interpreter.steps)

        term_no_eta = parse_lambda_string("λx.(x x)")
        interpreter._collect_used_vars(term_no_eta)
        self.assertIsNone(interpreter.eta_convert_once(term_no_eta))

        term_no_eta2 = parse_lambda_string("λx.((λy.x) x)")
        interpreter._collect_used_vars(term_no_eta2)
        self.assertIsNone(interpreter.eta_convert_once(term_no_eta2))

    # --- 测试解释器归约 ---
    def test_interpret_identity_app(self):
        interpreter = LambdaInterpreter(max_steps=10)
        term = parse_lambda_string("(λx.x) y")
        result = interpreter.interpret(term)
        self.assertLambdaEqual(Variable("y"), result)
        self.assertTrue(interpreter.steps[-1].strip().endswith("(Final Result)"),
                        f"Last step was: {interpreter.steps[-1]}")

    def test_interpret_normal_order_omega(self):
        interpreter = LambdaInterpreter(strategy=ReductionStrategy.NORMAL_ORDER, max_steps=5)
        term = parse_lambda_string("(λx.x x) (λx.x x)")
        result = interpreter.interpret(term)
        self.assertTrue("(max steps reached)" in interpreter.steps[-1],
                        f"Last step was: {interpreter.steps[-1]}")
        self.assertIsInstance(result, Application)

    def test_interpret_applicative_order_omega(self):
        interpreter = LambdaInterpreter(strategy=ReductionStrategy.APPLICATIVE_ORDER, max_steps=5)
        term = parse_lambda_string("(λx.x x) (λx.x x)")
        result = interpreter.interpret(term)
        self.assertTrue("(max steps reached)" in interpreter.steps[-1],
                        f"Last step was: {interpreter.steps[-1]}")

    def test_interpret_s_k_k_to_i(self):
        s_str = "λs_x.λs_y.λs_z.s_x s_z (s_y s_z)"
        k_str = "λk_a.λk_b.k_a"
        term_str = f"( ({s_str}) ({k_str}) ) ({k_str})"
        term = parse_lambda_string(term_str)
        interpreter = LambdaInterpreter(max_steps=50)
        result = interpreter.interpret(term)
        self.assertIsInstance(result, Abstraction)
        final_abs = result
        self.assertIsInstance(final_abs.body, Variable)  # type: ignore
        self.assertEqual(final_abs.var.name, final_abs.body.name)  # type: ignore

    def test_y_combinator(self):  # Simplified Y combinator test
        Y_COMBINATOR = parse_lambda_string("λy_f.(λy_x.y_f (y_x y_x)) (λy_x.y_f (y_x y_x))")
        CONST_FUNC_GEN = parse_lambda_string("λrec.λval.val")  # G = λrec val. val
        Y_CONST = Application(Y_COMBINATOR, CONST_FUNC_GEN)  # Y G

        interpreter_yconst = LambdaInterpreter(max_steps=60)  # Increased steps
        result_Y_CONST = interpreter_yconst.interpret(Y_CONST)

        # Expected: λval.val (or alpha-equivalent, e.g. λv_0.v_0)
        self.assertIsInstance(result_Y_CONST, Abstraction,
                              f"Expected Abstraction, got {type(result_Y_CONST)}: {result_Y_CONST.to_latex()}")
        final_abs_const = result_Y_CONST
        self.assertIsInstance(final_abs_const.body, Variable,
                              f"Expected body to be Variable, got {type(final_abs_const.body)}")  # type: ignore
        self.assertEqual(final_abs_const.var.name, final_abs_const.body.name,
                         "Expected var and body name to match in Id function")  # type: ignore




    # def test_church_factorial_two(self):
    #     # 定义邱奇数和相关函数
    #     # TRUE = λt.λf.t
    #     TRUE_CHURCH = parse_lambda_string("λt.λf.t")
    #     # FALSE = λt.λf.f
    #     FALSE_CHURCH = parse_lambda_string("λt.λf.f")
    #
    #     # IS_ZERO = λn.n (λx.FALSE) TRUE
    #     IS_ZERO_CHURCH = parse_lambda_string(f"λn.n (λx.{FALSE_CHURCH.to_latex()}) {TRUE_CHURCH.to_latex()}")
    #
    #     # ZERO = λf.λx.x
    #     ZERO_CHURCH = parse_lambda_string("λf.λx.x")
    #     # ONE = λf.λx.f x
    #     ONE_CHURCH = parse_lambda_string("λf.λx.f x")
    #     # TWO = λf.λx.f (f x)
    #     TWO_CHURCH = parse_lambda_string("λf.λx.f (f x)")
    #
    #     # SUCC = λn.λf.λx.f (n f x)
    #     SUCC_CHURCH = parse_lambda_string("λn.λf.λx.f (n f x)")
    #
    #     # PRED = λn.λf.λx.n (λg.λh.h (g f)) (λu.x) (λu.u)
    #     PRED_CHURCH = parse_lambda_string("λn.λf.λx.n (λg.λh.h (g f)) (λu.x) (λu.u)")
    #
    #     # MULT = λm.λn.λf.m (n f)  ( simplified: λm.λn.λf.λx. m (n f) x if we want full church num as result)
    #     # For MULT A B, it means apply A times the function (apply B times f)
    #     # MULT = λm.λn.λf. m (n f)
    #     MULT_CHURCH = parse_lambda_string("λmult_m.λmult_n.λmult_f. mult_m (mult_n mult_f)")
    #
    #     # Y Combinator: λg.(λx.g (x x)) (λx.g (x x))
    #     Y_COMBINATOR = parse_lambda_string("λy_g.(λy_x.y_g (y_x y_x)) (λy_x.y_g (y_x y_x))")
    #
    #     # FACT_HELPER = λfact.λn. (IS_ZERO n) ONE (MULT n (fact (PRED n)))
    #     # Using .to_latex() to embed complex terms into the string
    #     fact_helper_str = (
    #         f"λfact.λn. ({IS_ZERO_CHURCH.to_latex()} n) "
    #         f"({ONE_CHURCH.to_input_string()}) "
    #         f"(({MULT_CHURCH.to_input_string()} n) (fact ({PRED_CHURCH.to_input_string()} n)))"
    #     )
    #     FACT_HELPER = parse_lambda_string(fact_helper_str)
    #
    #     # FACTORIAL = Y_COMBINATOR FACT_HELPER
    #     FACTORIAL = Application(Y_COMBINATOR, FACT_HELPER)
    #
    #     # Test FACTORIAL TWO
    #     fact_two_term = Application(FACTORIAL, TWO_CHURCH)
    #
    #     # 阶乘归约需要非常多的步骤
    #     interpreter_factorial = LambdaInterpreter(max_steps=100000, strategy=ReductionStrategy.NORMAL_ORDER)
    #     lambda_logger.info(f"Testing Factorial TWO with term: {fact_two_term.to_latex()}")
    #
    #     result_fact_two = interpreter_factorial.interpret(fact_two_term)
    #     lambda_logger.info(f"Factorial TWO result: {result_fact_two.to_latex()}")
    #     lambda_logger.info(f"Number of steps for Factorial TWO: {len(interpreter_factorial.steps)}")
    #
    #     # 期望结果是 TWO_CHURCH (λf.λx.f (f x)) 或其 Alpha 等价形式
    #     # 由于 Alpha 等价的复杂性，直接比较对象可能失败。
    #     # 我们可以比较它们的 LaTeX 形式，或者尝试将它们都应用于相同的参数。
    #     # 一个简化的检查是看它是否是一个抽象，其内部结构是否类似。
    #
    #     # 为了使断言更可靠，我们将结果和期望的 TWO_CHURCH 都应用于一个通用函数和参数
    #     # 例如，应用到 (λz.z+1) 和 0 (在 Lambda 演算中表示)
    #     # 这超出了当前解释器的能力。
    #     # 因此，我们将依赖于结构和 LaTeX 形式的比较。
    #
    #     # 尝试直接比较 (可能会因 alpha-equivalence 失败)
    #     # self.assertLambdaEqual(TWO_CHURCH, result_fact_two, "Factorial TWO should reduce to TWO_CHURCH")
    #
    #     # 比较 LaTeX 形式 (更宽松，但仍可能因变量名不同而失败)
    #     # self.assertEqual(TWO_CHURCH.to_latex(), result_fact_two.to_latex())
    #
    #     # 更健壮的测试：检查归约后的项是否具有邱奇数2的结构
    #     # λf.λx.f (f x)
    #     self.assertIsInstance(result_fact_two, Abstraction, "Factorial result should be an Abstraction (λf...)")
    #     f_abs = result_fact_two
    #     self.assertIsInstance(f_abs.body, Abstraction, "Body of λf should be Abstraction (λx...)")  # type: ignore
    #     x_abs = f_abs.body  # type: ignore
    #
    #     # Body of λx should be f (f x) which is Application(f, Application(f, x))
    #     self.assertIsInstance(x_abs.body, Application, "Body of λx should be an Application (f (f x))")  # type: ignore
    #     app1 = x_abs.body  # type: ignore
    #
    #     self.assertIsInstance(app1.func, Variable, "Outer function in f(fx) should be f")  # type: ignore
    #     self.assertEqual(app1.func.name, f_abs.var.name,
    #                      "Outer function var name should match λf var name")  # type: ignore
    #
    #     self.assertIsInstance(app1.arg, Application, "Argument in f(fx) should be an Application (f x)")  # type: ignore
    #     app2 = app1.arg  # type: ignore
    #
    #     self.assertIsInstance(app2.func, Variable, "Inner function in (f x) should be f")  # type: ignore
    #     self.assertEqual(app2.func.name, f_abs.var.name,
    #                      "Inner function var name should match λf var name")  # type: ignore
    #
    #     self.assertIsInstance(app2.arg, Variable, "Innermost argument in (f x) should be x")  # type: ignore
    #     self.assertEqual(app2.arg.name, x_abs.var.name, "Innermost arg name should match λx var name")  # type: ignore

    # --- 测试装饰器 ---
    def test_input_validation_decorators_type_check(self):
        # Test LambdaInterpreter.__init__ type_check for 'strategy'
        with self.assertRaisesRegex(TypeError,
                                    r"Keyword argument 'strategy' of __init__ expected <enum 'ReductionStrategy'>, got <class 'str'>"):
            LambdaInterpreter(strategy="not_a_strategy")  # type: ignore

        # Test LambdaInterpreter.__init__ type_check for 'max_steps'
        with self.assertRaisesRegex(TypeError,
                                    r"Keyword argument 'max_steps' of __init__ expected <class 'int'>, got <class 'str'>"):
            LambdaInterpreter(max_steps="not_an_int")  # type: ignore

        # Test parse_lambda_string type_check for 's'
        with self.assertRaisesRegex(TypeError,
                                    r"Keyword argument 's' of parse_lambda_string expected <class 'str'>, got <class 'int'>"):
            parse_lambda_string(s=123)  # type: ignore

    def test_input_validation_decorators_non_empty_string(self):
        # Test parse_lambda_string's non_empty_string_check for 's'
        with self.assertRaisesRegex(ValueError,
                                    r"Argument 's' \('.*'\) of parse_lambda_string cannot be an empty or whitespace-only string\."):
            parse_lambda_string(s="")
        with self.assertRaisesRegex(ValueError,
                                    r"Argument 's' \('.*'\) of parse_lambda_string cannot be an empty or whitespace-only string\."):
            parse_lambda_string(s="   ")

        # Test Variable.__init__'s non_empty_string_check for 'name'
        with self.assertRaisesRegex(ValueError,
                                    r"Argument 'name' \('.*'\) of __init__ cannot be an empty or whitespace-only string\."):
            Variable(name=" ")

        # with self.assertRaisesRegex(ValueError,
        #                             r"Argument 'name' of __init__ cannot be an empty or whitespace-only string\."):
        #     Variable(name=" ")

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

