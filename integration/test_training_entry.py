"""Validate model-path propagation before any distributed/GPU initialization."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

class TrainingEntryTests(unittest.TestCase):
    def test_model_root_reaches_config_before_distributed_init(self):
        path = Path(__file__).resolve().parents[1] / 'wan_va/train.py'
        run = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'run')
        cfg = SimpleNamespace(wan22_pretrained_model_name_or_path='old-path')
        class StopBeforeGPU(Exception): pass
        def init(*args):
            self.assertEqual(cfg.wan22_pretrained_model_name_or_path, '/models/test-base')
            raise StopBeforeGPU
        ns = {'VA_CONFIGS': {'test': cfg}, 'os': os, 'json': json,
              'init_distributed': init}
        exec(compile(ast.Module(body=[run], type_ignores=[]), str(path), 'exec'), ns)
        with patch.dict(os.environ, {'MODEL_ROOT': '/models/test-base'}, clear=True):
            with self.assertRaises(StopBeforeGPU):
                ns['run'](SimpleNamespace(config_name='test'))

if __name__ == '__main__': unittest.main()
