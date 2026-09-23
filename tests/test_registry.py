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
