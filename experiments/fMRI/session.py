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
from copy import deepcopy
import time
import pylink
import numpy as np
import itertools
from PIL import Image
from psychopy import logging
import scipy.stats as ss
from scipy.stats import expon
from psychopy.visual import GratingStim, TextStim, Circle
from psychopy.core import getTime
from exptools2.core import Session, PylinkEyetrackerSession
from stimuli import FixationBullsEye, FixationCue, Gabors, Checkerboards
from trial import (
    TestTrial,
    TaskTrial_train,
    TaskTrial,
    PingTrial,
    RestingTrial,
    SuckerTrial,
    InstructionTrial,
    DummyWaiterTrial,
    OutroTrial,
    FeedbackTrial,
)

rng = random.SystemRandom()

class PredSession(PylinkEyetrackerSession):
    def __init__(
        self,
        output_str,
        output_dir,
        subject,
        ses_nr,
        task,
        run_nr,
        settings_file,
        eyetracker_on=True,
    ):
        """Initializes StroopSession object.

        Parameters
        ----------
        output_str : str
            Basename for all output-files (like logs), e.g., "sub-01_task-stroop_ses-1"
        output_dir : str
            Path to desired output-directory (default: None, which results in $pwd/logs)
        settings_file : str
            Path to yaml-file with settings (default: None, which results in the package's
            default settings file (in data/default_settings.yml)
        """
        super().__init__(
            output_str,
            output_dir=output_dir,
            settings_file=settings_file,
            eyetracker_on=eyetracker_on,
        )  # initialize parent class!

        self.subject = subject
        self.ses_nr = ses_nr
        self.task = task
        self.run_nr = run_nr
        self.data_yml_log = {}

        # Create log folder if it does not exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # set realtime mode for higher timing precision
        pylink.beginRealTimeMode(100)

        self._create_text_loading()
        self._create_stimuli()
        self._create_yaml_log()
        self.save_yaml_log()
        self.create_sequences()        
        self.create_trials()

        print("--------------------------------")
        print(
            "    /\\_/\\           ___\n   = o_o =_______    \\ \\ \n    __^      __(  \.__) )\n(@)<_____>__(_____)____/"
        )
        print("Author: @Ningkai Wang")
        print("--------------------------------")

    def create_sequences(self):
        """
        Creates all trials' parameters.

        variables:
        self.TD_pattern:
            - used to determine the target and distractor locations
            - each run includes 48 trials, 75% HPL are distractors
        self.TD_list:
            - used to determine the target and distractor locations
        self.oris_gabors:
            - used to determine the orientation of gabors
            - Target gabor is always tilted, distractor gabor is always horizontal or vertical
        self.ping_pairs:
            - used to determine the ping locations
        """

        # Create sequence of trials
        if self.ses_nr == 'practice':
            self.nr_task = 48
            self.nr_ping = 0
            self.nr_rest = 0
            self.nr_sucker = 0
            self.seq_trials = np.hstack([
                                        np.tile('TaskTrial', self.nr_task), 
                                        ])
        elif self.ses_nr == 'test':
            self.nr_task = 48
            self.nr_ping = 48
            self.nr_rest = 36
            self.nr_sucker = 24
            self.seq_trials = np.hstack([
                                        np.tile('TaskTrial', self.nr_task), 
                                        np.tile('PingTrial', self.nr_ping), 
                                        np.tile('RestingTrial', self.nr_rest), 
                                        np.tile('SuckerTrial', self.nr_sucker)
                                        ])
        elif self.ses_nr == 'train':
            if self.run_nr in [0,]:
                self.nr_task = 192
                self.nr_ping = 0
                self.nr_rest = 0
                self.nr_sucker = 0
                self.seq_trials = np.hstack([
                                        np.tile('TaskTrial', self.nr_task), 
                                        ])
            else:
                self.nr_task = 48
                self.nr_ping = 48
                self.nr_rest = 0
                self.nr_sucker = 0
                self.seq_trials = np.hstack([
                                        np.tile('TaskTrial', 48), 
                                        np.tile('PingTrial', 48),
                                        ])
        else:
            raise ValueError("session should be 'practice', 'train', or 'test'")
        np.random.shuffle(self.seq_trials)

        if self.task == 'neutral':
            self.HPL = None
        elif self.task == 'bias1':
            self.HPL = self.HPL_1
        elif self.task == 'bias2':
            self.HPL = self.HPL_2
        else: 
            raise ValueError("task should be 'neutral', 'bias1' or 'bias2'")

        # Create [target, distractor] pattern, each pattern includes 48 trials, 75% HPL are distractors
        self.TD_pattern = list(itertools.permutations(self.angles_gabors, 2))
        self.TD_pattern = np.tile(self.TD_pattern, (4, 1))
        if self.HPL is not None:
            for i in range(int(len(self.TD_pattern)/2)):
                if self.TD_pattern[i, 0] == self.HPL:
                    tmp = deepcopy(self.TD_pattern[i, 0])
                    self.TD_pattern[i, 0] = self.TD_pattern[i, 1]
                    self.TD_pattern[i, 1] = tmp
        print("Original TD_pattern")
        print(self.TD_pattern)

        self.TD_list = np.tile(self.TD_pattern, (int(self.nr_task/len(self.TD_pattern)), 1))
        np.random.shuffle(self.TD_list)
        
        # Create oritentation of gabors, can be perpendicular or tilted
        self.oris_gabors = np.empty((self.nr_task,2))
        oris_gabors_list_tt = np.hstack([np.repeat(45, int(self.oris_gabors.shape[0]/2)), np.repeat(135, self.oris_gabors.shape[0]-int(self.oris_gabors.shape[0]/2))])
        oris_gabors_list_pp = np.hstack([np.repeat(0, int(self.oris_gabors.shape[0]/2)), np.repeat(90, self.oris_gabors.shape[0]-int(self.oris_gabors.shape[0]/2))])
        np.random.shuffle(oris_gabors_list_tt)
        self.oris_gabors[:, 0] = deepcopy(oris_gabors_list_tt)
        np.random.shuffle(oris_gabors_list_pp)
        self.oris_gabors[:, 1] = deepcopy(oris_gabors_list_pp)

        # for i in range(len(self.TD_list)):
        #     self.oris_gabors[i, 0] = rng.choice([45, 135])
        #     self.oris_gabors[i, 1] = rng.choice([0, 180])

        # Create Ping trials, in each trial 1 pings are displayed. 
        self.seq_ping = np.empty((0,1))
        for i in range(int(self.nr_ping/len(self.angles_pings))):
            self.seq_ping = np.concatenate((self.seq_ping, 
                                              np.array(rng.sample(sorted(self.angles_pings), 
                                                         len(self.angles_pings)))[:, np.newaxis]), 
                                                         axis=0)
        self.oris_pings = np.hstack([np.repeat(0, int(self.seq_ping.shape[0]/2)), np.repeat(45, self.seq_ping.shape[0]-int(self.seq_ping.shape[0]/2))])
        np.random.shuffle(self.oris_pings)

    def _create_ping_pairs(self):
        """
        Select ping pairs from ping pool, the pings in each pair should be different and 90 degree apart.
        """
        ping_pairs = np.empty((0,2))
        print('Creating Ping pairs...')
        dist_pings = 90
        run = 0
        
        while len(ping_pairs)*2 != len(self.angles_pings):
            if run > 0:
                print('Creation failed, trying to re-run it, run ', run)
            run += 1
            ping_pool = deepcopy(self.angles_pings)
            ping_pool_tmp = deepcopy(ping_pool)
            ping_pairs = np.empty((0,2))
            t0 = getTime()

            while len(ping_pool) > 0:
                ping1 = rng.choice(ping_pool_tmp)
                ping_pool_tmp = np.delete(ping_pool_tmp, ping_pool_tmp==ping1)
                ping2 = rng.choice(ping_pool_tmp)
                ping_pool_tmp = np.delete(ping_pool_tmp, ping_pool_tmp==ping2)
                
                if np.abs(ping1-ping2) >= dist_pings and np.abs(ping1-ping2) <= 360-dist_pings:
                    ping_pairs = np.concatenate((ping_pairs, [[ping1, ping2]]))
                    ping_pool = deepcopy(ping_pool_tmp)
                else:
                    ping_pool_tmp = deepcopy(ping_pool)
                
                if len(ping_pool)==2 and (np.abs(ping_pool[0]-ping_pool[1]) <= dist_pings or 
                                          np.abs(ping_pool[0]-ping_pool[1]) >= 360-dist_pings):
                    break

                if getTime() - t0 > 0.5:
                    print('Time out, re-run it')
                    break
        
        ping_pairs = ping_pairs.astype(int)

        print('Ping pairs created successfully')

        return ping_pairs

    def _create_stimuli(self):
        """Creates all stimuli used in the experiment."""
        # create instruction text
        self.instruction_text = self.settings["stimuli"].get("instruction_text")
        self.instruction_text = (
            eval(f"f'{self.instruction_text}'")
        )
            
        # Create picture locations
        self._create_locations()

        # create fixation cross
        self._create_fixation()

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
            
        for _, (angle, ori) in enumerate(list(itertools.product(self.angles_gabors, np.array([0, 45, 90, 135]).astype(int)))):
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
        
        for _, (angle, ori) in enumerate(list(itertools.product(self.angles_pings, [0, 45, 135, 180]))):
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

        self.fsmask = Circle(
            win=self.win,
            units="deg",
            radius=50,
            pos=[0,0],
            edges=360,
            fillColor=0,
            lineColor=0,
        )
        self.fsmask.draw()

        self.win.flip()
        # time. sleep(5)
        self.win.flip()

    def create_trials(self):
        """Creates trials (ideally before running your session!)"""

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

        ## Create test trial
        # parameters  = {}
        # phase_durations = [0.4, 20, 2.4]
        # phase_names = ['fixation', 'stimulus', 'ITI']
        # keys = ['left', 'right']
        # self.trials.append(
        #                 TestTrial(
        #                     session=self,
        #                     trial_nr=self.trial_counter,
        #                     phase_durations=phase_durations,
        #                     phase_names=phase_names,
        #                     parameters=parameters,
        #                     keys=keys,
        #                     timing='seconds',
        #                     verbose=self.settings['monitor'].get("verbose"),
        #                     draw_each_frame=False,
        #                 )
        #             )

        # Create trials
        ind_TaskTrial = 0
        ind_PingTrial = 0
        self.resp_task = np.full(self.TD_list.shape[0], False) # record if the response is correct for each task trial
        self.resp_ping = np.empty(0) # record if the response is correct for each non-task trial
        # print('final TD_list')
        # print(self.TD_list)
        for trial_type in self.seq_trials:
            # Task trials
            if trial_type == 'TaskTrial':
                parameters  = {'trial_type': 'TaskTrial',
                               'angle_T': self.TD_list[ind_TaskTrial, 0], 
                               'ori_T': self.oris_gabors[ind_TaskTrial, 0], 
                               'angle_D': self.TD_list[ind_TaskTrial, 1], 
                               'ori_D': self.oris_gabors[ind_TaskTrial, 1],
                               'ind_TaskTrial': ind_TaskTrial,}

                if self.ses_nr == 'test':
                    phase_durations = [self.settings['stimuli'].get('fixdot_refresh_time'), 
                                    self.settings['stimuli'].get('stim_refresh_time'), 
                                    self.settings['stimuli'].get('ITI_time')]
                    phase_names = ['fixation', 'stimulus', 'ITI']

                    # setup keys
                    keys = self.settings['various'].get('buttons_test')
                    if self.oris_gabors[ind_TaskTrial, 0] == 135:
                        corr_key = self.settings['various'].get('buttons_test')[0]
                    elif self.oris_gabors[ind_TaskTrial, 0] == 45:
                        corr_key = self.settings['various'].get('buttons_test')[1]
                    else:
                        print(self.oris_gabors[ind_TaskTrial, 0])
                        raise ValueError("target location should be 45 or 135")

                    self.trials.append(
                            TaskTrial(
                                session=self,
                                trial_nr=self.trial_counter,
                                phase_durations=phase_durations,
                                phase_names=phase_names,
                                parameters=parameters,
                                keys=keys,
                                corr_key=corr_key,
                                timing='seconds',
                                verbose=self.settings['monitor'].get("verbose"),
                                draw_each_frame=False,
                            )
                        )
                elif self.ses_nr in ['practice', 'train']:
                    phase_durations = [self.settings['stimuli'].get('fixdot_refresh_time'), 
                                    self.settings['stimuli'].get('stim_refresh_time'), 
                                    self.settings['design'].get('resp_overtime'),
                                    self.settings['stimuli'].get('feedback_time')]
                    phase_names = ['fixation', 'stimulus', 'resp_overtime', 'feedback']

                    # setup keys
                    if self.oris_gabors[ind_TaskTrial, 0] == 135:
                        corr_key = self.settings['various'].get('buttons_train')[0]
                    elif self.oris_gabors[ind_TaskTrial, 0] == 45:
                        corr_key = self.settings['various'].get('buttons_train')[1]
                    else:
                        print(self.oris_gabors[ind_TaskTrial, 0])
                        raise ValueError("target location should be 45 or 135")
                    keys = self.settings['various'].get('buttons_train')

                    self.trials.append(
                            TaskTrial_train(
                                session=self,
                                trial_nr=self.trial_counter,
                                phase_durations=phase_durations,
                                phase_names=phase_names,
                                parameters=parameters,
                                keys=keys,
                                corr_key=corr_key,
                                timing='seconds',
                                verbose=self.settings['monitor'].get("verbose"),
                                draw_each_frame=False,
                            )
                        )
                ind_TaskTrial += 1
            # Ping trials
            elif trial_type == 'PingTrial':
                parameters  = {'trial_type': 'PingTrial',
                               'angle_Ping':self.seq_ping[ind_PingTrial ,0], 
                               'ori_Ping':self.oris_pings[ind_PingTrial],}
                keys = None
                if self.ses_nr == 'train':
                    phase_durations = [self.settings['stimuli'].get('fixdot_refresh_time'), 
                                    self.settings['stimuli'].get('stim_refresh_time'), 
                                    self.settings['design'].get('resp_overtime')]
                elif self.ses_nr == 'test':
                    phase_durations = [self.settings['stimuli'].get('fixdot_refresh_time'), 
                                    self.settings['stimuli'].get('stim_refresh_time'), 
                                    self.settings['stimuli'].get('ITI_time')]
                phase_names = ['fixation', 'stimulus', 'ITI']
                self.trials.append(
                        PingTrial(
                            session=self,
                            trial_nr=self.trial_counter,
                            phase_durations=phase_durations,
                            phase_names=phase_names,
                            parameters=parameters,
                            keys=keys,
                            timing='seconds',
                            verbose=self.settings['monitor'].get("verbose"),
                            draw_each_frame=False,
                        )
                    )
                ind_PingTrial += 1
            # Resting trials
            elif trial_type == 'RestingTrial':
                parameters = {'trial_type': 'RestingTrial',}
                phase_durations = [self.settings['stimuli'].get('fixdot_refresh_time'), 
                                   self.settings['stimuli'].get('stim_refresh_time'),
                                   self.settings['stimuli'].get('ITI_time')]
                phase_names = ['fixation', 'stimulus', 'ITI']
                self.trials.append(
                        RestingTrial(
                            session=self,
                            trial_nr=self.trial_counter,
                            phase_durations=phase_durations,
                            phase_names=phase_names,
                            parameters=parameters,
                            keys=None,
                            timing="seconds",
                            verbose=self.settings['monitor'].get("verbose"),
                            draw_each_frame=False,
                        )
                    )
            # Sucker trials
            elif trial_type == 'SuckerTrial':
                parameters = {'trial_type': 'SuckerTrial',}
                phase_durations = [self.settings['stimuli'].get('fixdot_refresh_time'), 
                                   self.settings['stimuli'].get('stim_refresh_time'), 
                                   self.settings['stimuli'].get('ITI_time')]
                phase_names = ['fixation', 'stimulus', 'ITI']
                self.trials.append(
                        SuckerTrial(
                            session=self,
                            trial_nr=self.trial_counter,
                            phase_durations=phase_durations,
                            phase_names=phase_names,
                            parameters=parameters,
                            keys=None,
                            timing="seconds",
                            verbose=self.settings['monitor'].get("verbose"),
                            draw_each_frame=False,
                        )
                    )
            else:
                raise ValueError("trial type should be TaskTrial, PingTrial, RestingTrial or SuckerTrial")
            
            self.trial_counter += 1

        self.trials.append(FeedbackTrial(session=self, trial_nr=self.trial_counter))

    def _create_locations(self):
        self.angles_gabors = [int(i) for i in np.linspace(45, 360+45, 4, endpoint=False)]%np.array([360])
        self.angles_pings = [int(i) for i in np.linspace(45, 360+45, 24, endpoint=False)]%np.array([360])

    def _create_fixation(self):
        self.fixation_cue = {}
        self.fixation_w = FixationBullsEye(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("fixation_size_pixels"),
            color=(0.5, 0.5, 0.5, 1),
            **{"lineWidth": self.settings["stimuli"].get("outer_fix_linewidth")},
        )

        self.fixation_dot = FixationCue(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("cue_size_deg"),
            color=-1,
            cross_lindwidth=self.settings["stimuli"].get("fixation_cross_lindwidth"),
            **{"lineWidth": self.settings["stimuli"].get("outer_fix_linewidth")},
        )

        self.fixation_dot.draw()

    def _create_text_loading(self):
        self.text_loading = TextStim(
            self.win,
            "Please wait a second, the experiment is loading...",
            height=self.settings["various"].get("text_height"),
            wrapWidth=self.settings["various"].get("text_width"),
            pos=[
                self.settings["various"].get("text_position_x"),
                self.settings["various"].get("text_position_y"),
            ],
            font="Arial",
            alignText="center",
            anchorHoriz="center",
            anchorVert="center",
        )
        self.text_loading.draw()
        self.win.flip()

    def _create_yaml_log(self):
        # every n block, use the new sequences and the new stimuli
        
        self.yml_log = os.path.join(
            self.output_dir,
            f"sub-{str(self.subject).zfill(2)}_log.yml",
        )

        # determine if there is a log file. If so, load it.
        if os.path.isfile(self.yml_log):
            with open(self.yml_log, "r") as ymlseqfile:
                try:
                    print(ymlseqfile)
                    print('XXXXXX')
                    yml_random = yaml.safe_load(ymlseqfile)
                except yaml.YAMLError as exc:
                    print(exc)
            print(yml_random)
            self.HPL_1 = yml_random.get("design").get("HPL_1")
            self.HPL_2 = yml_random.get("design").get("HPL_2")

        # if there is no sequence log file, create the HPLs from scratch
        else:
            location_pool = set(self.angles_gabors)
            # Only use neutral locations for the first session
            # Biased locations are used in the second session
            self.HPL_1 = rng.sample([*location_pool], 1)[0]
            location_pool.remove(self.HPL_1)
            # The second biased HPL is used in the third session
            self.HPL_2 = rng.sample([*location_pool], 1)[0]

            self.data_yml_log["design"] = {
                "HPL_1": int(self.HPL_1),
                "HPL_2": int(self.HPL_2),
            }

    def save_yaml_log(self):
        if not os.path.isfile(self.yml_log):
            with open(self.yml_log, "w") as ymlseqfile:
                yaml.dump(
                    self.data_yml_log, ymlseqfile, default_flow_style=False
                )

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

    def close(self):
        """Closes the experiment."""
        super().close()  # close parent class!
