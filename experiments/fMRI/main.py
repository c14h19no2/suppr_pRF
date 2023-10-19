#!/usr/bin/env python
#-*- coding: utf-8 -*-

from datetime import datetime
import argparse
from pathlib import Path
from psychopy import logging
from session import PredSession, PingSession, AwarenessSession

parser = argparse.ArgumentParser(description='A script to run the suppression-pRF task.')
parser.add_argument('subject', default=None, nargs='?', 
                    help='the subject of the experiment, as a zero-filled integer, such as 001, or 04.')
parser.add_argument('ses', default=None, type=str, nargs='?', 
                    help="the ses nr of the experiment, can be 'practice', 'train' or 'test'..")
parser.add_argument('task', default=None, type=str, nargs='?', 
                    help="the task of the experiment, can be 'neutral', 'bias1', 'bias2', 'ping', or 'AwarenessSession'.")
parser.add_argument('run', default=None, type=int, nargs='?', 
                    help='the run nr of the experiment, an integer, such as 1, or 99.')
parser.add_argument('eyelink', default=0, type=int, nargs='?')

cmd_args = parser.parse_args()
subject, ses, task, run, eyelink = cmd_args.subject, cmd_args.ses, cmd_args.task, cmd_args.run, cmd_args.eyelink

if task == '0':
    task = 'neutral'
elif task == '1':
    task = 'bias1'
elif task == '2':
    task = 'bias2'

# Check if the subject number is valid
if subject is None:
    raise ValueError(
        'subject must be a integer, such as 1, or 99')

if ses not in ['practice', 'train', 'test']:
    raise ValueError(
        'session must be a string, such as train, or test')

if task not in ['neutral', 'bias1', 'bias2', 'ping', 'awareness']:
    if task not in ['0', '1', '2']:
        raise ValueError(
            'task must be a string, such as neutral, bias1, or bias2, or 0, 1, or 2')

if not isinstance(run, int):
    raise ValueError(
        'run must be a integer, such as 1, or 99')

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
# output_str = f'sub-{str(subject).zfill(2)}_ses-{str(ses).zfill(2)}_task-pred_run-{str(run).zfill(2)}'
output_str = f'sub-{str(subject).zfill(2)}_ses-{str(ses).zfill(2)}_task-{task}_run-{str(run).zfill(2)}'
settings_fn = Path(__file__).parent / 'settings.yml'

# Adjust the output file name if exists
# if Path.exists(output_dir / output_str):
#     run += 1
#     output_str = f'sub-{str(subject).zfill(2)}_ses-{str(ses).zfill(2)}_task-pred_run-{str(run).zfill(2)}'

# Create the session object
if task == 'ping':
    session_object = PingSession(output_str=output_str,
                        output_dir=output_dir,
                        subject=subject,
                        ses_nr=ses,
                        task=task,
                        run_nr=run,
                        settings_file=settings_fn, 
                        eyetracker_on=eyetracker_on)
elif task in ['neutral', 'bias1', 'bias2']:
    session_object = PredSession(output_str=output_str,
                        output_dir=output_dir,
                        subject=subject,
                        ses_nr=ses,
                        task=task,
                        run_nr=run,
                        settings_file=settings_fn, 
                        eyetracker_on=eyetracker_on)
elif task == 'awareness':
    session_object = AwarenessSession(output_str=output_str,
                        output_dir=output_dir,
                        subject=subject,
                        ses_nr=ses,
                        task=task,
                        run_nr=run,
                        settings_file=settings_fn, 
                        eyetracker_on=eyetracker_on)

# Run the session
logging.warn(f'Writing results to: {Path(session_object.output_dir) / session_object.output_str}')
session_object.run()
session_object.close()