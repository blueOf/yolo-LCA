import ast
import unittest
from pathlib import Path


class TestSpatialMeanCenter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block_path = Path(__file__).resolve().parent / "ultralytics" / "nn" / "modules" / "block.py"
        cls.block_code = cls.block_path.read_text(encoding="utf-8")
        cls.block_ast = ast.parse(cls.block_code)
        cls.class_node = next(
            (n for n in cls.block_ast.body if isinstance(n, ast.ClassDef) and n.name == "SpatialMeanCenter"),
            None,
        )

    def test_class_exists(self):
        self.assertIsNotNone(self.class_node, "SpatialMeanCenter class should exist in block.py")

    def test_forward_uses_spatial_mean_subtraction(self):
        self.assertIsNotNone(self.class_node, "SpatialMeanCenter class should exist in block.py")
        forward = next(
            (n for n in self.class_node.body if isinstance(n, ast.FunctionDef) and n.name == "forward"),
            None,
        )
        self.assertIsNotNone(forward, "SpatialMeanCenter must implement forward")
        ret = next((n for n in forward.body if isinstance(n, ast.Return)), None)
        self.assertIsNotNone(ret, "forward should return centered tensor")
        self.assertIsInstance(ret.value, ast.BinOp)
        self.assertIsInstance(ret.value.op, ast.Sub)
        self.assertIsInstance(ret.value.left, ast.Name)
        self.assertEqual(ret.value.left.id, "x")

        mean_call = ret.value.right
        self.assertIsInstance(mean_call, ast.Call)
        self.assertIsInstance(mean_call.func, ast.Attribute)
        self.assertEqual(mean_call.func.attr, "mean")

        keywords = {k.arg: k.value for k in mean_call.keywords}
        self.assertIn("dim", keywords)
        self.assertIn("keepdim", keywords)

        dim_value = keywords["dim"]
        self.assertIsInstance(dim_value, ast.Tuple)
        dims = []
        for elt in dim_value.elts:
            if isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub) and isinstance(elt.operand, ast.Constant):
                dims.append(-elt.operand.value)
            elif isinstance(elt, ast.Constant):
                dims.append(elt.value)
        self.assertEqual(dims, [-2, -1])
        self.assertIsInstance(keywords["keepdim"], ast.Constant)
        self.assertIs(keywords["keepdim"].value, True)

    def test_tasks_imports_module_for_yaml_resolution(self):
        tasks_path = Path(__file__).resolve().parent / "ultralytics" / "nn" / "tasks.py"
        tasks_code = tasks_path.read_text(encoding="utf-8")
        self.assertIn("SpatialMeanCenter", tasks_code)


if __name__ == "__main__":
    unittest.main()
