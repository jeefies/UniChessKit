import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from unichess_kit.registry import EngineSpec, RegistryError, build_player_factory, load_object


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.added = []

    def tearDown(self):
        for name in [k for k in sys.modules if k.split(".")[0].startswith("kitreg_")]:
            del sys.modules[name]
        sys.path[:] = [p for p in sys.path if not p.startswith(str(self.dir))]
        shutil.rmtree(self.dir, ignore_errors=True)

    def make_pkg(self, root, name, body):
        pkg = self.dir / root / name
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(body, encoding="utf-8")
        return self.dir / root

    def test_load_from_root(self):
        root = self.make_pkg("eng", "kitreg_a", "def make(**kw):\n    return lambda: kw\n")
        f = build_player_factory(EngineSpec(factory="kitreg_a:make", root=str(root),
                                            kwargs={"x": 1}))
        self.assertEqual(f(), {"x": 1})

    def test_runtime_kwargs_passed_but_not_in_identity(self):
        """runtime（如临时服务目录）传给工厂，但不进 identity() → 不改配置哈希，续跑可换。"""
        root = self.make_pkg("rt", "kitreg_rt", "def make(**kw):\n    return lambda: kw\n")
        spec = EngineSpec(factory="kitreg_rt:make", root=str(root), kwargs={"x": 1},
                          runtime={"server_dir": "/tmp/a"})
        self.assertEqual(build_player_factory(spec)(), {"x": 1, "server_dir": "/tmp/a"})
        other = EngineSpec(factory="kitreg_rt:make", root=str(root), kwargs={"x": 1},
                           runtime={"server_dir": "/tmp/b"})
        self.assertEqual(spec.identity(), other.identity())
        self.assertNotIn("runtime", spec.identity())
        self.assertEqual(EngineSpec.from_dict(spec.to_dict()), spec)
        self.assertNotIn("runtime", EngineSpec(factory="a:b").to_dict())
        with self.assertRaisesRegex(RegistryError, "重复"):
            build_player_factory(EngineSpec(factory="kitreg_rt:make", root=str(root),
                                            kwargs={"x": 1}, runtime={"x": 2}))

    def test_same_top_level_from_other_root_rejected(self):
        """R/T 同名顶层包的老问题：第二个根目录的同名包必须报错，而不是静默复用第一个。"""
        r1 = self.make_pkg("one", "kitreg_core", "V = 1\n")
        r2 = self.make_pkg("two", "kitreg_core", "V = 2\n")
        self.assertEqual(load_object("kitreg_core:V", str(r1)), 1)
        with self.assertRaisesRegex(RegistryError, "冲突"):
            load_object("kitreg_core:V", str(r2))

    def test_module_outside_root_rejected(self):
        with self.assertRaisesRegex(RegistryError, "不在引擎根目录"):
            load_object("tabnanny:check", str(self.make_pkg("x", "kitreg_x", "")))

    def test_spi_version_checked(self):
        root = self.make_pkg("s", "kitreg_s", "KIT_SPI_VERSION = 999\nf = 1\n")
        with self.assertRaisesRegex(RegistryError, "KIT_SPI_VERSION"):
            load_object("kitreg_s:f", str(root))

    def test_errors(self):
        with self.assertRaises(RegistryError):
            load_object("no_colon")
        with self.assertRaises(RegistryError):
            load_object("json:nope")
        with self.assertRaises(RegistryError):
            load_object("json:dumps", str(self.dir / "missing"))
        with self.assertRaises(RegistryError):
            EngineSpec.from_dict({"factory": "a:b", "extra": 1})


if __name__ == "__main__":
    unittest.main()
