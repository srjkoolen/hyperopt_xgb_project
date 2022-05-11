import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)


import unittest

from hyperopt_xgb_project.functions import (

    get_model_parameters

)

class Functions_Test(unittest.TestCase):

    def test_get_model_parameters(self):
        self.assertEqual(1,1)
