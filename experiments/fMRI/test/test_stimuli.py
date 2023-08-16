#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
print(sys.path)
import os
import glob
import random
import yaml
import math
import h5py
from copy import deepcopy
import urllib.request
import time
import pylink
import numpy as np
import itertools
from PIL import Image
from psychopy import logging
import scipy.stats as ss
from scipy.stats import expon
from psychopy.visual import GratingStim, ImageStim, TextStim
from psychopy.core import getTime
from psychopy import parallel
from exptools2.core import Session, PylinkEyetrackerSession
from stimuli import FixationBullsEye, FixationCue, PlaceHolderCircles, Gabors, Checkerboards
from trial import (
    TaskTrial,
    PingTrial,
    RestingTrial,
    SuckerTrial,
    InstructionTrial,
    DummyWaiterTrial,
    OutroTrial,
    FeedbackTrial,
)
from psychopy import sound




class testSession(PylinkEyetrackerSession):
    def __init__(
        self,
        output_str,
        output_dir,
        subject,
        ses_nr,
        run_nr,
        settings_file,
        eyetracker_on=True,
    ):
        super().__init__(
            output_str,
            output_dir=output_dir,
            settings_file=settings_file,
            eyetracker_on=eyetracker_on,
        )  # initialize parent class!

        try:
            self.port = parallel.ParallelPort(address=0x0378)
            self.port.setData(0)
            self.parallel_triggering = True
        except:
            logging.warn(f"Attempted import of Parallel Port failed")
            self.parallel_triggering = False

        self.subject = subject
        self.ses_nr = ses_nr
        self.run_nr = run_nr
        self.data_yml_random_log = {}
        print(self.win.monitorFramePeriod)
        print('WWWWW')
        self.create_stimuli()
        self.create_trials()
        


    def create_stimuli(self):
        self.fixation_w = FixationBullsEye(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("fixation_size_pixels"),
            color=(0.5, 0.5, 0.5, 1),
            **{"lineWidth": self.settings["stimuli"].get("outer_fix_linewidth")},
        )
        self.fixation_w.draw()

        self.fixation_dot = FixationCue(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("cue_size_pixels"),
            color=-1,
            **{"lineWidth": self.settings["stimuli"].get("outer_fix_linewidth")},
        )
        self.fixation_dot.draw()

        self.fixation_flicker = FixationCue(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("cue_size_pixels"),
            color=-1,
            **{"lineWidth": self.settings["stimuli"].get("outer_fix_linewidth")},
        )
        self.fixation_flicker.draw()

        self.empty_circles = PlaceHolderCircles(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("stim_size_deg"),
            color=-0.5,
            ecc=self.settings["stimuli"].get("distance_from_center"),
            linewidth=self.settings["stimuli"].get("stim_circle_linewidth"),
            fill=False,
        )
        self.empty_circles.draw()

        angles_gabors = [int(i) for i in np.linspace(45, 360+45, 4, endpoint=False)]%np.array([360])
        angles_pings = [int(i) for i in np.linspace(45, 360+45, 24, endpoint=False)]%np.array([360])
        oris_gabors = [int(i) for i in np.linspace(0, 180, 4, endpoint=False)]

        # Set up gabors
        self.gabors = {}
        self.gabors['test'] = Gabors(
            win=self.win, 
            size=self.settings["stimuli"].get("stim_size_deg"), 
            sf=8, 
            ori=45, 
            ecc=self.settings["stimuli"].get("distance_from_center"), 
            angle=0,
            phase=0, 
            contrast=1,
            units="deg")
            
        for i, (angle, ori) in enumerate(list(itertools.product(angles_gabors, oris_gabors))):
            # print(angle, ori)
            self.gabors[(angle, ori)] = Gabors(
                win=self.win, 
                size=self.settings["stimuli"].get("stim_size_deg"), 
                sf=8, 
                ori=ori, 
                ecc=self.settings["stimuli"].get("distance_from_center"), 
                angle=angle,
                phase=0, 
                contrast=1,
                units="deg")
            self.gabors[(angle, ori)].draw()
        
            
        # Set up checkerboards
        self.checkerboards = {}
        self.checkerboards['test'] = Checkerboards(
            win=self.win,
            size=self.settings["stimuli"].get("stim_size_deg"), 
            sf=8, 
            ori=45, 
            ecc=self.settings["stimuli"].get("distance_from_center"), 
            angle=45,
            phase=0, 
            contrast=1,
            units="deg")
        
        for i, (angle, ori) in enumerate(list(itertools.product(angles_pings, oris_gabors))):
            # print(angle, ori)
            self.checkerboards[(angle, ori)] = Checkerboards(
                win=self.win, 
                size=self.settings["stimuli"].get("stim_size_deg"), 
                sf=8, 
                ori=ori, 
                ecc=self.settings["stimuli"].get("distance_from_center"), 
                angle=angle,
                phase=0, 
                contrast=1,
                units="deg")
            self.checkerboards[(angle, ori)].draw()
        self.gabors['test'].draw()
        self.checkerboards['test'].draw()
        self.win.flip()
        self.win.flip()



    def create_trials(self):
        """Creates trials (ideally before running your session!)"""
        
        self.instruction_text = self.settings["stimuli"].get("instruction_text")
        instruction_trial = InstructionTrial(
            session=self,
            trial_nr=0,
            phase_durations=[np.inf],
            txt=self.instruction_text,
            keys=["space"],
            draw_each_frame=False,
        )

        dummy_trial = DummyWaiterTrial(
            session=self,
            trial_nr=1,
            phase_durations=[np.inf, self.settings["design"].get("start_duration")],
            txt=self.settings["stimuli"].get("pretrigger_text"),
            draw_each_frame=False,
        )
        self.trials = [instruction_trial, dummy_trial]
        self.trial_counter = 2
        
        # Create trials
        

        # Gabor Trial
        parameters  = {'angle_1':45, 'ori_1':135, 'angle_2':315, 'ori_2':45}
        self.trials.append(
                TaskTrial(
                    session=self,
                    trial_nr=self.trial_counter,
                    phase_durations=[0.4, 2, 2],
                    phase_names=[
                        'fixation',
                        'stimulus',
                        'ITI',
                        ],
                    parameters=parameters,
                    keys=[
                        'left', 'right'
                        ],
                    timing='seconds',
                    verbose=self.settings['monitor'].get("verbose"), ############################## CHANGE TO FALSE
                    draw_each_frame=False, ############################## CHANGE TO FALSE
                )
            )
        
        # Ping Trial
        self.trial_counter += 1
        parameters   = {'angle_1':45, 'ori_1':45, 'angle_2':90, 'ori_2':90}
        self.trials.append(
                PingTrial(
                    session=self,
                    trial_nr=self.trial_counter,
                    phase_durations=[0.4 ,2 , 2],
                    phase_names=[
                        'fixation',
                        'stimulus',
                        'ITI',
                        ],
                    parameters=parameters,
                    keys=[
                        "left", "right"
                        ],
                    timing="seconds",
                    verbose=self.settings['monitor'].get("verbose"),
                    draw_each_frame=False,
                )
            )
        
        # Resting Trial
        self.trial_counter += 1
        parameters = {}
        self.trials.append(
                RestingTrial(
                    session=self,
                    trial_nr=self.trial_counter,
                    phase_durations=[0.4 ,2, 2],
                    phase_names=[
                        'fixation',
                        'stimulus',
                        'ITI',
                        ],
                    parameters=parameters,
                    keys=[
                        "left", "right"
                        ],
                    timing="seconds",
                    verbose=self.settings['monitor'].get("verbose"),
                    draw_each_frame=False,
                )
            )

        # Sucker Trial
        self.trial_counter += 1
        parameters  = {'angle_1':45, 'ori_1':135, 'angle_2':315, 'ori_2':45}
        self.trials.append(
                SuckerTrial(
                    session=self,
                    trial_nr=self.trial_counter,
                    phase_durations=[0.4, 2, 2],
                    phase_names=[
                        'fixation',
                        'stimulus',
                        'ITI',
                        ],
                    parameters=parameters,
                    keys=[
                        'left', 'right'
                        ],
                    timing='seconds',
                    verbose=self.settings['monitor'].get("verbose"),
                    draw_each_frame=False,
                )
            )

    def parallel_trigger(self, trigger):
        if self.parallel_triggering:
            self.port.setData(trigger)
            time.sleep(self.settings["design"].get("ttl_trigger_delay"))
            self.port.setData(0)
            time.sleep(self.settings["design"].get("ttl_trigger_delay"))
            # P = windll.inpoutx64
            # P.Out32(0x0378, self.settings['design'].get('ttl_trigger_blank')) # send the event code (could be 1-20)
            # time.sleep(self.settings['design'].get('ttl_trigger_delay')) # wait for 1 ms for receiving the code
            # P.Out32(0x0378, 0) # send a code to clear the register
            # time.sleep(self.settings['design'].get('ttl_trigger_delay')) # wait for 1 ms"""
        else:
            logging.warn(f"Would have sent trigger {trigger}")

    def run(self):
        """Runs experiment."""
        # self.create_trials()  # create them *before* running!

        if self.eyetracker_on:
            self.calibrate_eyetracker()

        self.start_experiment()

        if self.eyetracker_on:
            self.start_recording_eyetracker()
        for trial in self.trials:
            trial.run()

        self.close()

# run the session
subject = 99
ses = 1
run = 1
output_dir = Path(__file__).parent / 'logs' / f'sub-{str(subject).zfill(2)}'

if not Path.exists(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
output_str = f'sub-{str(subject).zfill(2)}_ses-{str(ses).zfill(2)}_task-pred_run-{str(run).zfill(2)}'
settings_fn = Path(__file__).parent.parent / 'settings.yml'
eyetracker_on = False

session_object = testSession(output_str=output_str,
                    output_dir=output_dir,
                    subject=subject,
                    ses_nr=ses,
                    run_nr=run,
                    settings_file=settings_fn, 
                    eyetracker_on=eyetracker_on)

session_object.run()

