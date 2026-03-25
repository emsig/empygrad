# Copyright 2016 The emsig community.
#
# This file is part of empygrad.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.

# Import all modules
from empygrad import model
from empygrad import utils
from empygrad import kernel
from empygrad import scripts

# Import most important functions
from empygrad.model import bipole, dipole, loop, ip_and_q
from empygrad.utils import EMArray, set_minimum, get_minimum, Report

# For top-namespace
from empygrad.scripts import fdesign, tmtemod
from empygrad.model import analytical, gpr, dipole_k, fem, tem

__all__ = ['model', 'utils', 'kernel', 'scripts',
           'bipole', 'dipole', 'ip_and_q', 'EMArray', 'set_minimum',
           'get_minimum', 'Report']

# Version defined in utils, so we can easier use it within the package itself.
__version__ = utils.__version__
