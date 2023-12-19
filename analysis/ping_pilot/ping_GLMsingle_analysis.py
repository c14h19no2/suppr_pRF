import os
from pathlib import Path
import numpy as np
import argparse
import yaml
from ping_GLMsingle import fit_GLMsingle

parser = argparse.ArgumentParser(description='GLMsingle setup')
parser.add_argument('yml_config', default=None, nargs='?', 
                    help='yml config file path')

cmd_args = parser.parse_args()
yml_config = cmd_args.yml_config

"""Exp parameters setup
"""
if os.path.isfile(yml_config):
    with open(yml_config, "r") as ymlfile:
        try:
            opt = yaml.safe_load(ymlfile)
        except yaml.YAMLError as exc:
            print(exc)

design_opt = opt['EXP_opt']['design']
path_opt = opt['EXP_opt']['path']
GLMsingle_opt = opt['GLMsingle_opt']

# set exp parameters
design_opt['runs'] = np.array(design_opt['runs'])
design_opt['angles'] = [*range(0, 360, int(360/design_opt['angles_nr']))]
design_opt['total_time'] = (design_opt['TR_nr']-design_opt['blank_TR_nr'])*design_opt['TR']

# set path
path_opt['datadir'] = Path(path_opt['datadir'])
path_opt['datadir_bids'] = Path(path_opt['datadir'], 'bids_data')
path_opt['datadir_fmriprep'] = Path(path_opt['datadir'], 'derivatives', 'fmriprep')
path_opt['datadir_freesufer'] = Path(path_opt['datadir'], 'derivatives', 'freesurfer')
path_opt['outputdir'] = Path(path_opt['datadir'], 'derivatives', 'GLMsingle_test', opt['EXP_opt']['name'], opt['EXP_opt']['path']['outputfolder'])
path_opt['figuredir'] = Path(path_opt['datadir'], 'derivatives', 'GLMsingle_test', opt['EXP_opt']['name'], opt['EXP_opt']['path']['figurefolder'])

# want retinamap?
output_typeC_retinamap = opt['output_opt']['output_typeC_retinamap']
output_typeD_retinamap = opt['output_opt']['output_typeD_retinamap']
"""GLMsingle parameters setup
"""

fit_GLMsingle(design_opt, path_opt, GLMsingle_opt, output_typeC_retinamap, output_typeD_retinamap)