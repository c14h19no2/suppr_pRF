#!/usr/bin/env python
#-*- coding: utf-8 -*-

import argparse
from pathlib import Path
from psychopy import logging
from session import RollDownTheWindowSession

parser = argparse.ArgumentParser(description='A script to run the suppression-pRF task.')
parser.add_argument('subject', default=None, nargs='?', 
                    help='the subject of the experiment, as a zero-filled integer, such as 001, or 04.')
parser.add_argument('eyelink', default=0, type=int, nargs='?')

cmd_args = parser.parse_args()
subject, eyelink = cmd_args.subject, cmd_args.eyelink

task = 'roll_down_the_window'

# Check if the subject number is valid
if isinstance(subject, int):
    raise ValueError(
        'subject must be a integer, such as 1, or 99')

# Determine whether to use the eyetracker
if eyelink == 1:
    eyetracker_on = True
    logging.warn("Using eyetracker")
else:
    eyetracker_on = False
    logging.warn("Using NO eyetracker")

# Make sure the output directory exists
output_dir = Path(__file__).parent / 'logs' / f'sub-{str(subject).zfill(2)}'
if not Path.exists(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
output_str = f'sub-{str(subject).zfill(2)}_task-{task}'
settings_fn = Path(__file__).parent / 'settings.yml'

# Create the session object
session_object = RollDownTheWindowSession(output_str=output_str,
                    output_dir=output_dir,
                    subject=subject,
                    task=task,
                    settings_file=settings_fn, 
                    eyetracker_on=eyetracker_on)

# Run the session
logging.warn(f'Writing results to: {Path(session_object.output_dir) / session_object.output_str}')
session_object.run()
session_object.close()
