import os
import sys

if __package__ is None or __package__ == "":
	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.diagnostic as d
import inspect
print(inspect.getsource(d.answer_with_local_hf_model).count("truncation_side"))