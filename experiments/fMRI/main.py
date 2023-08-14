#!/usr/bin/env python
#-*- coding: utf-8 -*-

from datetime import datetime
import argparse
import os.path as op
from os import listdir, makedirs
from psychopy import logging
from session import PredSession, LocalizerSession

parser = argparse.ArgumentParser(description='A categorily and spatitally prediction experiment')
parser.add_argument('subject', default=None, nargs='?', 
                    help='the subject of the experiment, as a zero-filled integer, such as 001, or 04.')
parser.add_argument('ses', default=None, type=int, nargs='?', 
                    help='the ses nr of the experimental ses, an integer, such as 1, or 99.')
parser.add_argument('run', default=None, type=int, nargs='?', 
                    help='the run nr of the experimental ses, an integer, such as 1, or 99.')
parser.add_argument('eyelink', default=0, type=int, nargs='?')

cmd_args = parser.parse_args()
subject, ses, run, eyelink = cmd_args.subject, cmd_args.ses, cmd_args.run, cmd_args.eyelink

if subject is None:
    subject = 99
    # subject = datetime.now().strftime("%y%m%d%H")

if ses is None:
    raise ValueError(
        'ses_nr must be 0 (training full sequence), 1 (partial sequence), or 2 (violate sequence)')
    # subject = datetime.now().strftime("%y%m%d%H")

if run is None:
    raise ValueError(
        'run must be a integer, such as 1, or 99')

if eyelink == 1:
    eyetracker_on = True
    logging.warn("Using eyetracker")
else:
    eyetracker_on = False
    logging.warn("Using NO eyetracker")

output_dir = op.join(op.dirname(__file__), 'logs', f'sub-{str(subject).zfill(2)}')
if not op.exists(output_dir):
    makedirs(output_dir)
output_str = f'sub-{str(subject).zfill(2)}_ses-{str(ses).zfill(2)}_task-pred_run-{str(run).zfill(2)}'
settings_fn = op.join(op.dirname(__file__), 'settings.yml')

while any([output_str in f for f in listdir(output_dir)]):
    run += 1
    output_str = f'sub-{str(subject).zfill(2)}_ses-{str(ses).zfill(2)}_task-pred_run-{str(run).zfill(2)}'

if ses in [0, 1, 2]:
    session_object = PredSession(output_str=output_str,
                        output_dir=output_dir,
                        subject=subject,
                        ses_nr=ses,
                        run_nr=run,
                        settings_file=settings_fn, 
                        eyetracker_on=eyetracker_on)
elif ses == 3:
    session_object = LocalizerSession(output_str=output_str,
                        output_dir=output_dir,
                        subject=subject,
                        ses_nr=ses,
                        settings_file=settings_fn, 
                        eyetracker_on=eyetracker_on)
else:
    raise ValueError(
        'ses_nr must be 0 (training full sequence), 1 (partial sequence), 2 (violate sequence), or 3 (localizer session)')

logging.warn(f'Writing results to: {op.join(session_object.output_dir, session_object.output_str)}')
session_object.run()
session_object.close()