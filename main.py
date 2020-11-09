import csv
import os
import subprocess

import numpy as np
from scipy.special import logsumexp

from semiring import LogSemiring, ProbabilitySemiring
from util import CTMWriter, read_htk, wer

htk_file = "lattice.1.htk"

file_name_hyp = "test_file.ctm"
file_name_ref = "transcriptions.stm"
output_folder = "wer_tests"

lattice = read_htk(htk_file)

scored_lattice = lattice.forward_backward(LogSemiring())