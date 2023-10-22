#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent
sys.path.append(str(parent_dir))
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
from psychopy.tools import monitorunittools
from exptools2.core import Session, PylinkEyetrackerSession
from stimuli import (
    FixationBullsEye,
    FixationDot,
    FixationDot_flk,
    Gabors,
    Checkerboards,
    CheckerboardsAdjContrast,
    PlaceHolder,
    Highlighter,
)
from trial import (
    TestTrial,
    TaskTrial_train,
    TaskTrial,
    PingTrial,
    RestingTrial,
    SuckerTrial,
    InstructionTrial,
    DummyWaiterTrial,
    WaitStartTriggerTrial,
    FeedbackTrial,
    RollDownTheWindowTrial,
    PingpRFTrial,
    PingpRFTrial_train,
    InstructionTrial_awareness,
    AwarenessCheckTrial,
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
        self._create_yaml_log()
        self._create_stimuli()
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
        if self.ses_nr == "practice":
            self.nr_task = self.settings["design"].get("supprpRF_practice_task_nr")
            self.nr_ping = 0
            self.nr_rest = 0
            self.nr_sucker = 0

        elif self.ses_nr == "test":
            self.nr_task = self.settings["design"].get("supprpRF_task_nr")
            self.nr_ping = self.settings["design"].get("supprpRF_ping_nr")
            self.nr_rest = self.settings["design"].get("supprpRF_rest_nr")
            self.nr_sucker = self.settings["design"].get("supprpRF_sucker_nr")

        elif self.ses_nr == "train":
            if self.run_nr in [
                0,
            ]:
                self.nr_task = self.settings["design"].get("supprpRF_train_task_nr")
                self.nr_ping = 0
                self.nr_rest = 0
                self.nr_sucker = 0

            else:
                self.nr_task = self.settings["design"].get("supprpRF_testtrain_task_nr")
                self.nr_ping = self.settings["design"].get("supprpRF_testtrain_ping_nr")
                self.nr_rest = 0
                self.nr_sucker = 0

        else:
            raise ValueError("session should be 'practice', 'train', or 'test'")

        self.seq_trials = np.hstack(
            [
                np.tile("TaskTrial", self.nr_task),
                np.tile("PingTrial", self.nr_ping),
                np.tile("RestingTrial", self.nr_rest),
                np.tile("SuckerTrial", self.nr_sucker),
            ]
        )

        np.random.shuffle(self.seq_trials)

        if self.task == "neutral":
            self.HPL = None
        elif self.task == "bias1":
            self.HPL = self.HPL_1
        elif self.task == "bias2":
            self.HPL = self.HPL_2
        else:
            raise ValueError("task should be 'neutral', 'bias1' or 'bias2'")

        # Create [target, distractor] pattern, each pattern includes 48 trials, 75% HPL are distractors
        self.TD_pattern = list(itertools.permutations(self.angles_gabors, 2))
        self.TD_pattern = np.tile(self.TD_pattern, (4, 1))
        if self.HPL is not None:
            for i in range(int(len(self.TD_pattern) / 2)):
                if self.TD_pattern[i, 0] == self.HPL:
                    tmp = deepcopy(self.TD_pattern[i, 0])
                    self.TD_pattern[i, 0] = self.TD_pattern[i, 1]
                    self.TD_pattern[i, 1] = tmp

        self.TD_list = np.tile(
            self.TD_pattern, (int(self.nr_task / len(self.TD_pattern)), 1)
        )
        np.random.shuffle(self.TD_list)

        # Create oritentation of gabors, can be perpendicular or tilted
        self.oris_gabors = np.empty((self.nr_task, 2))
        oris_gabors_list_tt = np.hstack(
            [
                np.repeat(45, int(self.oris_gabors.shape[0] / 2)),
                np.repeat(
                    135, self.oris_gabors.shape[0] - int(self.oris_gabors.shape[0] / 2)
                ),
            ]
        )
        oris_gabors_list_pp = np.hstack(
            [
                np.repeat(0, int(self.oris_gabors.shape[0] / 2)),
                np.repeat(
                    90, self.oris_gabors.shape[0] - int(self.oris_gabors.shape[0] / 2)
                ),
            ]
        )
        np.random.shuffle(oris_gabors_list_tt)
        self.oris_gabors[:, 0] = deepcopy(oris_gabors_list_tt)
        np.random.shuffle(oris_gabors_list_pp)
        self.oris_gabors[:, 1] = deepcopy(oris_gabors_list_pp)

        # for i in range(len(self.TD_list)):
        #     self.oris_gabors[i, 0] = rng.choice([45, 135])
        #     self.oris_gabors[i, 1] = rng.choice([0, 180])

        # Create Ping trials, in each trial 1 pings are displayed.
        self.seq_ping = np.empty((0, 1))
        for i in range(int(self.nr_ping / len(self.angles_pings))):
            self.seq_ping = np.concatenate(
                (
                    self.seq_ping,
                    np.array(
                        rng.sample(sorted(self.angles_pings), len(self.angles_pings))
                    )[:, np.newaxis],
                ),
                axis=0,
            )
        self.oris_pings = np.hstack(
            [
                np.repeat(0, int(self.seq_ping.shape[0] / 2)),
                np.repeat(45, self.seq_ping.shape[0] - int(self.seq_ping.shape[0] / 2)),
            ]
        )
        np.random.shuffle(self.oris_pings)

    def _create_ping_pairs(self):
        """
        Select ping pairs from ping pool, the pings in each pair should be different and 90 degree apart.
        """
        ping_pairs = np.empty((0, 2))
        logging.warn("Creating Ping pairs...")
        dist_pings = 90
        run = 0

        while len(ping_pairs) * 2 != len(self.angles_pings):
            if run > 0:
                logging.warn("Creation failed, trying to re-run it, run ", run)
            run += 1
            ping_pool = deepcopy(self.angles_pings)
            ping_pool_tmp = deepcopy(ping_pool)
            ping_pairs = np.empty((0, 2))
            t0 = getTime()

            while len(ping_pool) > 0:
                ping1 = rng.choice(ping_pool_tmp)
                ping_pool_tmp = np.delete(ping_pool_tmp, ping_pool_tmp == ping1)
                ping2 = rng.choice(ping_pool_tmp)
                ping_pool_tmp = np.delete(ping_pool_tmp, ping_pool_tmp == ping2)

                if (
                    np.abs(ping1 - ping2) >= dist_pings
                    and np.abs(ping1 - ping2) <= 360 - dist_pings
                ):
                    ping_pairs = np.concatenate((ping_pairs, [[ping1, ping2]]))
                    ping_pool = deepcopy(ping_pool_tmp)
                else:
                    ping_pool_tmp = deepcopy(ping_pool)

                if len(ping_pool) == 2 and (
                    np.abs(ping_pool[0] - ping_pool[1]) <= dist_pings
                    or np.abs(ping_pool[0] - ping_pool[1]) >= 360 - dist_pings
                ):
                    break

                if getTime() - t0 > 0.5:
                    logging.warn("Time out, re-run it")
                    break

        ping_pairs = ping_pairs.astype(int)

        logging.warn("Ping pairs created successfully")

        return ping_pairs

    def _create_stimuli(self):
        """Creates all stimuli used in the experiment."""
        # create instruction text
        self.instruction_text = self.settings["stimuli"].get("instruction_text")

        # Create picture locations
        self._create_locations()

        # create fixation cross
        self._create_fixation()
        # Set up gabors
        self.gabors = {}
        self.gabors["test"] = Gabors(
            win=self.win,
            size=self.settings["stimuli"].get("stim_size_deg"),
            sf=self.settings["stimuli"].get("stim_spatial_freq"),
            ori=45,
            ecc=self.settings["stimuli"].get("distance_from_center"),
            roll_dist=self.roll_dist,
            angle=0,
            phase=0,
            contrast=1,
            units="deg",
        )

        for _, (angle, ori) in enumerate(
            list(
                itertools.product(
                    self.angles_gabors, np.array([0, 45, 90, 135]).astype(int)
                )
            )
        ):
            self.gabors[(angle, ori)] = Gabors(
                win=self.win,
                size=self.settings["stimuli"].get("stim_size_deg"),
                sf=self.settings["stimuli"].get("stim_spatial_freq"),
                ori=ori,
                ecc=self.settings["stimuli"].get("distance_from_center"),
                roll_dist=self.roll_dist,
                angle=angle,
                phase=0,
                contrast=1,
                units="deg",
            )
            self.gabors[(angle, ori)].draw()

        # Set up checkerboards
        self.checkerboards = {}
        self.checkerboards["test"] = Checkerboards(
            win=self.win,
            size=self.settings["stimuli"].get("stim_size_deg"),
            sf=self.settings["stimuli"].get("stim_spatial_freq"),
            ori=45,
            ecc=self.settings["stimuli"].get("distance_from_center"),
            roll_dist=self.roll_dist,
            angle=45,
            phase=0,
            contrast=1,
            temporal_freq=self.settings["stimuli"].get("fixation_temporal_freq"),
            units="deg",
        )

        for _, (angle, ori) in enumerate(
            list(itertools.product(self.angles_pings, [0, 45, 135, 180]))
        ):
            self.checkerboards[(angle, ori)] = Checkerboards(
                win=self.win,
                size=self.settings["stimuli"].get("stim_size_deg"),
                sf=self.settings["stimuli"].get("stim_spatial_freq"),
                ori=ori,
                ecc=self.settings["stimuli"].get("distance_from_center"),
                roll_dist=self.roll_dist,
                angle=angle,
                phase=0,
                contrast=1,
                temporal_freq=self.settings["stimuli"].get("fixation_temporal_freq"),
                units="deg",
            )
            self.checkerboards[(angle, ori)].draw()

        self.gabors["test"].draw()
        self.checkerboards["test"].draw()

        self.fsmask = Circle(
            win=self.win,
            units="deg",
            radius=50,
            pos=[0, 0],
            edges=360,
            fillColor=self.settings["window"].get("color"),
            lineColor=self.settings["window"].get("color"),
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
            txt_height=self.settings["various"].get("text_height"),
            txt_width=self.settings["various"].get("text_width"),
            txt_position_x=self.settings["various"].get("text_position_x"),
            txt_position_y=self.settings["various"].get("text_position_y")
            + self.roll_dist,
            draw_each_frame=False,
        )
        if self.ses_nr == "practice":
            dummy_txt = self.settings["stimuli"].get("pretrigger_text")
        if self.ses_nr == "train" and (not self.settings["design"].get("mri_scan")):
            dummy_txt = self.settings["stimuli"].get("pretrigger_text")
        elif self.ses_nr == "train" and self.settings["design"].get("mri_scan"):
            dummy_txt = self.settings["stimuli"].get("pretrigger_text")
        elif self.ses_nr == "test" and (not self.settings["design"].get("mri_scan")):
            dummy_txt = self.settings["stimuli"].get("pretrigger_text")
        elif self.ses_nr == "test" and self.settings["design"].get("mri_scan"):
            dummy_txt = ""
        else:
            raise ValueError("session should be 'practice', 'train', or 'test'")

        dummy_trial = DummyWaiterTrial(
            session=self,
            trial_nr=0,
            phase_durations=[np.inf, self.settings["design"].get("start_duration")],
            txt=dummy_txt,
            draw_each_frame=False,
            txt_height=self.settings["various"].get("text_height"),
            txt_width=self.settings["various"].get("text_width"),
            txt_position_x=self.settings["various"].get("text_position_x"),
            txt_position_y=self.settings["various"].get("text_position_y"),
        )

        start_trial = WaitStartTriggerTrial(
            session=self,
            trial_nr=1,
            phase_durations=[np.inf],
            draw_each_frame=False,
        )

        if (not self.settings["design"].get("mri_scan")) or (self.ses_nr == "train"):
            self.trials = [instruction_trial]
            self.trials.append(
                DummyWaiterTrial(
                    session=self,
                    trial_nr=1,
                    phase_durations=[
                        np.inf,
                        self.settings["design"].get("train_start_duration"),
                    ],
                    txt=dummy_txt,
                    txt_height=self.settings["various"].get("text_height"),
                    txt_width=self.settings["various"].get("text_width"),
                    txt_position_x=self.settings["various"].get("text_position_x"),
                    txt_position_y=self.settings["various"].get("text_position_y"),
                    draw_each_frame=False,
                )
            )
        else:
            self.trials = [dummy_trial, start_trial]

        self.nr_instruction_trials = len(self.trials)
        self.trial_counter = len(self.trials)

        # # Create test trial
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
        self.resp_task = np.full(
            self.TD_list.shape[0], False
        )  # record if the response is correct for each task trial
        self.resp_ping = np.empty(
            0
        )  # record if the response is correct for each non-task trial

        for trial_type in self.seq_trials:
            # Task trials
            if trial_type == "TaskTrial":
                parameters = {
                    "trial_type": "TaskTrial",
                    "angle_T": self.TD_list[ind_TaskTrial, 0],
                    "ori_T": self.oris_gabors[ind_TaskTrial, 0],
                    "angle_D": self.TD_list[ind_TaskTrial, 1],
                    "ori_D": self.oris_gabors[ind_TaskTrial, 1],
                    "ind_TaskTrial": ind_TaskTrial,
                }

                if self.ses_nr == "test":
                    if self.settings["design"].get("mri_scan"):
                        if (
                            self.trial_counter + 1 - self.nr_instruction_trials
                        ) % 4 != 0:
                            last_phase_duration = self.settings["design"].get(
                                "ITI_time"
                            )
                        else:
                            last_phase_duration = np.inf
                    else:
                        last_phase_duration = self.settings["design"].get("ITI_time")
                    phase_durations = [
                        self.settings["design"].get("fixation_refresh_time"),
                        self.settings["design"].get("stim_refresh_time"),
                        last_phase_duration,
                    ]
                    phase_names = ["fixation", "stimulus", "ITI"]

                    # setup keys
                    if self.oris_gabors[ind_TaskTrial, 0] == 135:
                        corr_key = self.settings["various"].get("buttons_test")[0]
                    elif self.oris_gabors[ind_TaskTrial, 0] == 45:
                        corr_key = self.settings["various"].get("buttons_test")[1]
                    else:
                        logging.warn(
                            "Angle of target location is ",
                            self.oris_gabors[ind_TaskTrial, 0],
                        )
                        raise ValueError("target location should be 45 or 135")
                    keys = self.settings["various"].get("buttons_test")
                    parameters["corr_key"] = corr_key

                    self.trials.append(
                        TaskTrial(
                            session=self,
                            trial_nr=self.trial_counter,
                            phase_durations=phase_durations,
                            phase_names=phase_names,
                            parameters=parameters,
                            keys=keys,
                            timing="seconds",
                            verbose=self.settings["monitor"].get("verbose"),
                            draw_each_frame=False,
                        )
                    )
                elif self.ses_nr in ["practice", "train"]:
                    phase_durations = [
                        self.settings["design"].get("fixation_refresh_time"),
                        self.settings["design"].get("stim_refresh_time"),
                        self.settings["design"].get("resp_overtime"),
                        self.settings["design"].get("feedback_time"),
                    ]
                    phase_names = ["fixation", "stimulus", "resp_overtime", "feedback"]

                    # setup keys
                    if self.oris_gabors[ind_TaskTrial, 0] == 135:
                        corr_key = self.settings["various"].get("buttons_train")[0]
                    elif self.oris_gabors[ind_TaskTrial, 0] == 45:
                        corr_key = self.settings["various"].get("buttons_train")[1]
                    else:
                        logging.warn(
                            "Angle of target location is ",
                            self.oris_gabors[ind_TaskTrial, 0],
                        )
                        raise ValueError("target location should be 45 or 135")
                    keys = self.settings["various"].get("buttons_train")
                    parameters["corr_key"] = corr_key

                    self.trials.append(
                        TaskTrial_train(
                            session=self,
                            trial_nr=self.trial_counter,
                            phase_durations=phase_durations,
                            phase_names=phase_names,
                            parameters=parameters,
                            keys=keys,
                            timing="seconds",
                            verbose=self.settings["monitor"].get("verbose"),
                            draw_each_frame=False,
                        )
                    )
                ind_TaskTrial += 1
            # Ping trials
            elif trial_type == "PingTrial":
                parameters = {
                    "trial_type": "PingTrial",
                    "angle_Ping": self.seq_ping[ind_PingTrial, 0],
                    "ori_Ping": self.oris_pings[ind_PingTrial],
                }
                keys = None
                if self.ses_nr == "train":
                    phase_durations = [
                        self.settings["design"].get("fixation_refresh_time"),
                        self.settings["design"].get("stim_refresh_time"),
                        self.settings["design"].get("resp_overtime"),
                    ]
                elif self.ses_nr == "test":
                    if self.settings["design"].get("mri_scan"):
                        if (
                            self.trial_counter + 1 - self.nr_instruction_trials
                        ) % 4 != 0:
                            last_phase_duration = self.settings["design"].get(
                                "ITI_time"
                            )
                        else:
                            last_phase_duration = np.inf
                    else:
                        last_phase_duration = self.settings["design"].get("ITI_time")

                    phase_durations = [
                        self.settings["design"].get("fixation_refresh_time"),
                        self.settings["design"].get("stim_refresh_time"),
                        last_phase_duration,
                    ]
                phase_names = ["fixation", "stimulus", "ITI"]
                self.trials.append(
                    PingTrial(
                        session=self,
                        trial_nr=self.trial_counter,
                        phase_durations=phase_durations,
                        phase_names=phase_names,
                        parameters=parameters,
                        keys=keys,
                        timing="seconds",
                        verbose=self.settings["monitor"].get("verbose"),
                        draw_each_frame=False,
                    )
                )
                ind_PingTrial += 1
            # Resting trials
            elif trial_type == "RestingTrial":
                parameters = {
                    "trial_type": "RestingTrial",
                }
                if self.settings["design"].get("mri_scan"):
                    if (self.trial_counter + 1 - self.nr_instruction_trials) % 4 != 0:
                        last_phase_duration = self.settings["design"].get("ITI_time")
                    else:
                        last_phase_duration = np.inf
                else:
                    last_phase_duration = self.settings["design"].get("ITI_time")

                phase_durations = [
                    self.settings["design"].get("fixation_refresh_time"),
                    self.settings["design"].get("stim_refresh_time"),
                    last_phase_duration,
                ]
                phase_names = ["fixation", "stimulus", "ITI"]
                self.trials.append(
                    RestingTrial(
                        session=self,
                        trial_nr=self.trial_counter,
                        phase_durations=phase_durations,
                        phase_names=phase_names,
                        parameters=parameters,
                        keys=None,
                        timing="seconds",
                        verbose=self.settings["monitor"].get("verbose"),
                        draw_each_frame=False,
                    )
                )
            # Sucker trials
            elif trial_type == "SuckerTrial":
                parameters = {
                    "trial_type": "SuckerTrial",
                }
                if self.settings["design"].get("mri_scan"):
                    if (self.trial_counter + 1 - self.nr_instruction_trials) % 4 != 0:
                        last_phase_duration = self.settings["design"].get("ITI_time")
                    else:
                        last_phase_duration = np.inf
                else:
                    last_phase_duration = self.settings["design"].get("ITI_time")

                phase_durations = [
                    self.settings["design"].get("fixation_refresh_time"),
                    self.settings["design"].get("stim_refresh_time"),
                    last_phase_duration,
                ]
                phase_names = ["fixation", "stimulus", "ITI"]
                self.trials.append(
                    SuckerTrial(
                        session=self,
                        trial_nr=self.trial_counter,
                        phase_durations=phase_durations,
                        phase_names=phase_names,
                        parameters=parameters,
                        keys=None,
                        timing="seconds",
                        verbose=self.settings["monitor"].get("verbose"),
                        draw_each_frame=False,
                    )
                )
            else:
                raise ValueError(
                    "trial type should be TaskTrial, PingTrial, RestingTrial or SuckerTrial"
                )

            self.trial_counter += 1

        if (self.ses_nr == "test") and self.settings["design"].get("mri_scan"):
            self.trials.append(
                DummyWaiterTrial(
                    session=self,
                    trial_nr=self.trial_counter,
                    phase_durations=[
                        np.inf,
                        self.settings["design"].get("end_duration"),
                    ],
                    txt=dummy_txt,
                    draw_each_frame=False,
                )
            )
            self.trial_counter += 1
        else:
            self.trials.append(
                FeedbackTrial(session=self, trial_nr=self.trial_counter, keys=["space"])
            )
            self.trial_counter += 1

    def _create_locations(self):
        self.angles_gabors = [
            int(i) for i in np.linspace(45, 360 + 45, 4, endpoint=False)
        ] % np.array([360])
        self.angles_pings = [
            int(i) for i in np.linspace(45, 360 + 45, 24, endpoint=False)
        ] % np.array([360])

    def _create_fixation(self):
        self.fixbullseye = FixationBullsEye(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("distance_from_center"),
            color=(0, 0, 0, 1),
            pos=[0, self.roll_dist],
            **{
                "lineWidth": monitorunittools.deg2pix(
                    self.settings["stimuli"].get("outer_fix_linewidth"),
                    self.win.monitor,
                )
            },
        )

        self.fixation_dot = FixationDot(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("fixation_size_deg"),
            pos=[0, self.roll_dist],
            dotcolor=-1,
            linecolor=self.settings["window"].get("color"),
            cross_lindwidth=monitorunittools.deg2pix(
                self.settings["stimuli"].get("fixation_size_deg") / 5, self.win.monitor
            ),
        )

        self.fixation_dot.draw()

        self.fixation_dot_flk = FixationDot_flk(
            win=self.win,
            freq=self.settings["stimuli"].get("fixation_temporal_freq"),
            circle_radius=self.settings["stimuli"].get("fixation_size_deg"),
            pos=[0, self.roll_dist],
            dotcolor=-1,
            linecolor=self.settings["window"].get("color"),
            cross_lindwidth=monitorunittools.deg2pix(
                self.settings["stimuli"].get("fixation_size_deg") / 5, self.win.monitor
            ),
        )

        self.fixation_dot_flk.draw()

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
            units="deg",
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
                    yml_random = yaml.safe_load(ymlseqfile)
                except yaml.YAMLError as exc:
                    print(exc)
            self.data_yml_log = yml_random
            if yml_random.get("design") is not None:
                self.HPL_1 = yml_random.get("design").get("HPL_1")
                self.HPL_2 = yml_random.get("design").get("HPL_2")
                create_new_HPL = False
            else:
                create_new_HPL = True
            if yml_random.get("window") is not None:
                get_roll_dist = True
            else:
                get_roll_dist = False

        else:
            create_new_HPL = True
            get_roll_dist = False

        # if there is no HPL log file, create the HPLs from scratch
        if create_new_HPL:
            location_pool = set(
                [int(i) for i in np.linspace(45, 360 + 45, 4, endpoint=False)]
                % np.array([360])
            )
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
        if get_roll_dist:
            self.roll_dist = yml_random.get("window").get("roll_dist")
        else:
            self.roll_dist = 0

        self.data_yml_log["window"] = {"roll_dist": self.roll_dist}
        logging.warn(self.data_yml_log)

    def save_yaml_log(self):
        if not os.path.isfile(self.yml_log):
            with open(self.yml_log, "w") as ymlseqfile:
                yaml.dump(self.data_yml_log, ymlseqfile, default_flow_style=False)
        else:
            with open(self.yml_log, "r") as ymlseqfile:
                try:
                    yml_random = yaml.safe_load(ymlseqfile)
                except yaml.YAMLError as exc:
                    print(exc)

            if ("design" not in yml_random) or ("window" not in yml_random):
                with open(self.yml_log, "w") as ymlseqfile:
                    yaml.dump(self.data_yml_log, ymlseqfile, default_flow_style=False)

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


class RollDownTheWindowSession(PylinkEyetrackerSession):
    def __init__(
        self,
        output_str,
        output_dir,
        subject,
        task,
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
        self.task = task
        self.data_yml_log = {}

        # Create log folder if it does not exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # set realtime mode for higher timing precision
        pylink.beginRealTimeMode(100)

        self._create_yaml_log()
        self._create_fixation()
        self.create_trials()
        self.save_yaml_log()

        print("--------------------------------")
        print(
            "    /\\_/\\           ___\n   = o_o =_______    \\ \\ \n    __^      __(  \.__) )\n(@)<_____>__(_____)____/"
        )
        print("Author: @Ningkai Wang")
        print("--------------------------------")

    def _create_fixation(self):
        self.fixbullseye = FixationBullsEye(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("distance_from_center"),
            color=(0, 0, 0, 1),
            pos=[0, self.roll_dist],
            **{
                "lineWidth": monitorunittools.deg2pix(
                    self.settings["stimuli"].get("outer_fix_linewidth"),
                    self.win.monitor,
                )
            },
        )

        self.fixation_dot = FixationDot(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("fixation_size_deg"),
            pos=[0, self.roll_dist],
            dotcolor=-1,
            linecolor=self.settings["window"].get("color"),
            cross_lindwidth=monitorunittools.deg2pix(
                self.settings["stimuli"].get("fixation_size_deg") / 5, self.win.monitor
            ),
        )

    def create_trials(self):
        self.trial_counter = 0
        self.trials = []
        self.trials.append(
            RollDownTheWindowTrial(
                session=self,
                trial_nr=self.trial_counter,
                phase_durations=[np.inf],
                keys=self.settings["various"].get("buttons_test"),
                draw_each_frame=False,
            )
        )
        self.trial_counter += 1
        pass

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
                    yml_random = yaml.safe_load(ymlseqfile)
                except yaml.YAMLError as exc:
                    print(exc)
            logging.warn("Subject yaml log file loaded:" + str(yml_random))
            self.data_yml_log = yml_random
            if yml_random.get("window") is None:
                create_new_roll_dist = True
            else:
                self.roll_dist = yml_random.get("window").get("roll_dist")
                if self.roll_dist is None:
                    create_new_roll_dist = True
                else:
                    create_new_roll_dist = False
        else:
            create_new_roll_dist = True

        if create_new_roll_dist:
            self.roll_dist = 0
            self.data_yml_log["window"] = {
                "roll_dist": self.roll_dist,
            }

    def save_yaml_log(self):
        logging.warn("running save_yaml_log")
        if not os.path.isfile(self.yml_log):
            logging.warn("no yml log file")
            with open(self.yml_log, "w") as ymlseqfile:
                yaml.dump(self.data_yml_log, ymlseqfile, default_flow_style=False)
        else:
            with open(self.yml_log, "r") as ymlseqfile:
                try:
                    yml_random = yaml.safe_load(ymlseqfile)
                except yaml.YAMLError as exc:
                    print(exc)

            if "window" not in yml_random:
                logging.warn("window not in ymlseqfile")
                logging.warn(self.data_yml_log)
                with open(self.yml_log, "w") as ymlseqfile:
                    yaml.dump(self.data_yml_log, ymlseqfile, default_flow_style=False)
            elif yml_random.get("window").get("roll_dist") != self.roll_dist:
                logging.warn("roll_dist changed")
                with open(self.yml_log, "w") as ymlseqfile:
                    yaml.dump(self.data_yml_log, ymlseqfile, default_flow_style=False)
            else:
                logging.warn("roll_dist not changed")
                pass

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


class PingSession(PylinkEyetrackerSession):
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
        self._create_yaml_log()
        self._create_stimuli()
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
        self.ping_pairs:
            - used to determine the ping locations
        """

        # Create sequence of trials
        self.nr_ping = self.settings["design"].get("pingpRF_ping_nr")
        self.nr_rest = 0
        self.nr_sucker = 0
        self.seq_trials = np.hstack(
            [
                np.tile("PingTrial", self.nr_ping),
            ]
        )
        np.random.shuffle(self.seq_trials)
        self.resp_task = np.full(self.nr_ping, False)

        # Create Ping trials, in each trial 1 pings are displayed.
        self.seq_ping = np.empty((0, 1))
        for i in range(int(self.nr_ping / len(self.angles_pings))):
            self.seq_ping = np.concatenate(
                (
                    self.seq_ping,
                    np.array(
                        rng.sample(sorted(self.angles_pings), len(self.angles_pings))
                    )[:, np.newaxis],
                ),
                axis=0,
            )
            self.seq_ping_direction = np.tile([-1, 1], int(self.nr_ping / 2))
        self.oris_pings = np.hstack(
            [
                np.repeat(0, int(self.seq_ping.shape[0] / 2)),
                np.repeat(45, self.seq_ping.shape[0] - int(self.seq_ping.shape[0] / 2)),
            ]
        )
        np.random.shuffle(self.oris_pings)
        np.random.shuffle(self.seq_ping_direction)

    def _create_ping_pairs(self):
        """
        Select ping pairs from ping pool, the pings in each pair should be different and 90 degree apart.
        """
        ping_pairs = np.empty((0, 2))
        logging.warn("Creating Ping pairs...")
        dist_pings = 90
        run = 0

        while len(ping_pairs) * 2 != len(self.angles_pings):
            if run > 0:
                logging.warn("Creation failed, trying to re-run it, run ", run)
            run += 1
            ping_pool = deepcopy(self.angles_pings)
            ping_pool_tmp = deepcopy(ping_pool)
            ping_pairs = np.empty((0, 2))
            t0 = getTime()

            while len(ping_pool) > 0:
                ping1 = rng.choice(ping_pool_tmp)
                ping_pool_tmp = np.delete(ping_pool_tmp, ping_pool_tmp == ping1)
                ping2 = rng.choice(ping_pool_tmp)
                ping_pool_tmp = np.delete(ping_pool_tmp, ping_pool_tmp == ping2)

                if (
                    np.abs(ping1 - ping2) >= dist_pings
                    and np.abs(ping1 - ping2) <= 360 - dist_pings
                ):
                    ping_pairs = np.concatenate((ping_pairs, [[ping1, ping2]]))
                    ping_pool = deepcopy(ping_pool_tmp)
                else:
                    ping_pool_tmp = deepcopy(ping_pool)

                if len(ping_pool) == 2 and (
                    np.abs(ping_pool[0] - ping_pool[1]) <= dist_pings
                    or np.abs(ping_pool[0] - ping_pool[1]) >= 360 - dist_pings
                ):
                    break

                if getTime() - t0 > 0.5:
                    logging.warn("Time out, re-run it")
                    break

        ping_pairs = ping_pairs.astype(int)

        logging.warn("Ping pairs created successfully")

        return ping_pairs

    def _create_stimuli(self):
        """Creates all stimuli used in the experiment."""
        # create instruction text
        self.instruction_text = self.settings["stimuli"].get("instruction_text")
        self.instruction_text = eval(f"f'{self.instruction_text}'")

        # Create picture locations
        self._create_locations()

        # create fixation cross
        self._create_fixation()

        # Set up checkerboards

        contrast_range = self.settings["stimuli"].get(
            "ping_contrast_highest"
        ) - self.settings["stimuli"].get("ping_contrast_lowest")
        self.ping_contrast_adj_rate = contrast_range / self.settings["design"].get(
            "fixation_refresh_time"
        )

        self.checkerboards = {}

        for _, (angle, ori, direction) in enumerate(
            list(itertools.product(self.angles_pings, [0, 45, 135, 180], [-1, 1]))
        ):
            if direction == 1:
                base_contrast = self.settings["stimuli"].get("ping_contrast_lowest")
            elif direction == -1:
                base_contrast = self.settings["stimuli"].get("ping_contrast_highest")
            else:
                raise ValueError("direction should be 1 or -1")

            self.checkerboards[(angle, ori, direction)] = CheckerboardsAdjContrast(
                win=self.win,
                size=self.settings["stimuli"].get("stim_size_deg"),
                sf=self.settings["stimuli"].get("stim_spatial_freq"),
                ori=ori,
                ecc=self.settings["stimuli"].get("distance_from_center"),
                roll_dist=self.roll_dist,
                angle=angle,
                direction=direction,
                adj_rate=self.ping_contrast_adj_rate,
                phase=0,
                contrast=base_contrast,
                temporal_freq=self.settings["stimuli"].get("ping_temporal_freq"),
                units="deg",
            )
            self.checkerboards[(angle, ori, direction)].draw()

        self.fsmask = Circle(
            win=self.win,
            units="deg",
            radius=50,
            pos=[0, 0],
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
            txt_height=self.settings["various"].get("text_height"),
            txt_width=self.settings["various"].get("text_width"),
            txt_position_x=self.settings["various"].get("text_position_x"),
            txt_position_y=self.settings["various"].get("text_position_y")
            + self.roll_dist,
            draw_each_frame=False,
        )

        if self.ses_nr == "train":
            dummy_txt = self.settings["stimuli"].get("pretrigger_text")
        else:
            dummy_txt = ""

        dummy_trial = DummyWaiterTrial(
            session=self,
            trial_nr=1,
            phase_durations=[np.inf, self.settings["design"].get("start_duration")],
            txt=dummy_txt,
            draw_each_frame=False,
        )

        start_trial = WaitStartTriggerTrial(
            session=self,
            trial_nr=1,
            phase_durations=[np.inf],
            draw_each_frame=False,
        )

        if (not self.settings["design"].get("mri_scan")) or (self.ses_nr == "train"):
            self.trials = [instruction_trial]
            self.trials.append(
                DummyWaiterTrial(
                    session=self,
                    trial_nr=1,
                    phase_durations=[
                        np.inf,
                        self.settings["design"].get("train_start_duration"),
                    ],
                    txt=dummy_txt,
                    txt_height=self.settings["various"].get("text_height"),
                    txt_width=self.settings["various"].get("text_width"),
                    txt_position_x=self.settings["various"].get("text_position_x"),
                    txt_position_y=self.settings["various"].get("text_position_y")
                    + self.roll_dist,
                    draw_each_frame=False,
                )
            )
        else:
            self.trials = [dummy_trial, start_trial]

        self.nr_instruction_trials = len(self.trials)
        self.trial_counter = len(self.trials)

        # Create trials
        ind_PingTrial = 0
        for trial_type in self.seq_trials:
            # Ping trials
            if trial_type == "PingTrial":
                if self.ses_nr == "train":
                    keys = self.settings["various"].get("buttons_train")
                    if self.seq_ping_direction[ind_PingTrial] == 1:
                        corr_key = self.settings["various"].get("buttons_train")[0]
                    elif self.seq_ping_direction[ind_PingTrial] == -1:
                        corr_key = self.settings["various"].get("buttons_train")[1]
                    else:
                        raise ValueError("direction should be 1 or -1")

                    parameters = {
                        "trial_type": "PingTrial",
                        "angle_Ping": self.seq_ping[ind_PingTrial, 0],
                        "ori_Ping": self.oris_pings[ind_PingTrial],
                        "direction": self.seq_ping_direction[ind_PingTrial],
                        "corr_key": corr_key,
                        "ind_TaskTrial": ind_PingTrial,
                    }
                    phase_durations = [
                        self.settings["design"].get("fixation_refresh_time"),
                        self.settings["design"].get("stim_refresh_time"),
                        self.settings["design"].get("resp_overtime"),
                        self.settings["design"].get("feedback_time"),
                    ]
                    phase_names = ["fixation", "stimulus", "resp_overtime", "feedback"]

                    self.trials.append(
                        PingpRFTrial_train(
                            session=self,
                            trial_nr=self.trial_counter,
                            phase_durations=phase_durations,
                            phase_names=phase_names,
                            parameters=parameters,
                            keys=keys,
                            timing="seconds",
                            verbose=self.settings["monitor"].get("verbose"),
                            draw_each_frame=False,
                        )
                    )

                elif self.ses_nr == "test":
                    keys = self.settings["various"].get("buttons_test")
                    if self.seq_ping_direction[ind_PingTrial] == 1:
                        corr_key = self.settings["various"].get("buttons_test")[0]
                    elif self.seq_ping_direction[ind_PingTrial] == -1:
                        corr_key = self.settings["various"].get("buttons_test")[1]
                    else:
                        logging.warn(
                            "The direction of the change of contrast of ping is ",
                            self.seq_ping_direction[ind_PingTrial],
                        )
                        raise ValueError("direction should be 1 or -1")

                    parameters = {
                        "trial_type": "PingTrial",
                        "angle_Ping": self.seq_ping[ind_PingTrial, 0],
                        "ori_Ping": self.oris_pings[ind_PingTrial],
                        "direction": self.seq_ping_direction[ind_PingTrial],
                        "corr_key": corr_key,
                        "ind_TaskTrial": ind_PingTrial,
                    }

                    if self.settings["design"].get("mri_scan"):
                        if (
                            self.trial_counter + 1 - self.nr_instruction_trials
                        ) % 4 != 0:
                            last_phase_duration = self.settings["design"].get(
                                "ITI_time"
                            )
                        else:
                            last_phase_duration = np.inf
                    else:
                        last_phase_duration = self.settings["design"].get("ITI_time")

                    phase_durations = [
                        self.settings["design"].get("fixation_refresh_time"),
                        self.settings["design"].get("stim_refresh_time"),
                        last_phase_duration,
                    ]
                    phase_names = ["fixation", "stimulus", "ITI"]
                    self.trials.append(
                        PingpRFTrial(
                            session=self,
                            trial_nr=self.trial_counter,
                            phase_durations=phase_durations,
                            phase_names=phase_names,
                            parameters=parameters,
                            keys=keys,
                            timing="seconds",
                            verbose=self.settings["monitor"].get("verbose"),
                            draw_each_frame=False,
                        )
                    )
                ind_PingTrial += 1
            else:
                raise ValueError(
                    "trial type should be PingTrial, RestingTrial or SuckerTrial"
                )

            self.trial_counter += 1

        if (self.ses_nr == "test") and self.settings["design"].get("mri_scan"):
            self.trials.append(
                DummyWaiterTrial(
                    session=self,
                    trial_nr=self.trial_counter,
                    phase_durations=[
                        np.inf,
                        self.settings["design"].get("end_duration"),
                    ],
                    txt=dummy_txt,
                    draw_each_frame=False,
                )
            )
            self.trial_counter += 1
        else:
            self.trials.append(
                FeedbackTrial(session=self, trial_nr=self.trial_counter, keys=["space"])
            )
            self.trial_counter += 1

    def _create_locations(self):
        self.angles_pings = [
            int(i) for i in np.linspace(45, 360 + 45, 24, endpoint=False)
        ] % np.array([360])

    def _create_fixation(self):
        self.fixbullseye = FixationBullsEye(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("distance_from_center"),
            color=(0, 0, 0, 1),
            pos=[0, self.roll_dist],
            **{
                "lineWidth": monitorunittools.deg2pix(
                    self.settings["stimuli"].get("outer_fix_linewidth"),
                    self.win.monitor,
                )
            },
        )

        self.fixation_dot = FixationDot(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("fixation_size_deg"),
            pos=[0, self.roll_dist],
            dotcolor=-1,
            linecolor=self.settings["window"].get("color"),
            cross_lindwidth=monitorunittools.deg2pix(
                self.settings["stimuli"].get("fixation_size_deg") / 5, self.win.monitor
            ),
        )

        self.fixation_dot.draw()

        self.fixation_dot_flk = FixationDot_flk(
            win=self.win,
            freq=self.settings["stimuli"].get("fixation_temporal_freq"),
            circle_radius=self.settings["stimuli"].get("fixation_size_deg"),
            pos=[0, self.roll_dist],
            dotcolor=-1,
            linecolor=self.settings["window"].get("color"),
            cross_lindwidth=monitorunittools.deg2pix(
                self.settings["stimuli"].get("fixation_size_deg") / 5, self.win.monitor
            ),
        )

        self.fixation_dot_flk.draw()

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
            units="deg",
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
                    yml_random = yaml.safe_load(ymlseqfile)
                except yaml.YAMLError as exc:
                    print(exc)
            self.data_yml_log = yml_random
            if yml_random.get("window") is not None:
                get_roll_dist = True
            else:
                get_roll_dist = False

        else:
            get_roll_dist = False

        if get_roll_dist:
            self.roll_dist = yml_random.get("window").get("roll_dist")
        else:
            self.roll_dist = 0

        self.data_yml_log["window"] = {"roll_dist": self.roll_dist}

    def save_yaml_log(self):
        if not os.path.isfile(self.yml_log):
            with open(self.yml_log, "w") as ymlseqfile:
                yaml.dump(self.data_yml_log, ymlseqfile, default_flow_style=False)
        else:
            with open(self.yml_log, "r") as ymlseqfile:
                try:
                    yml_random = yaml.safe_load(ymlseqfile)
                except yaml.YAMLError as exc:
                    print(exc)

            if ("design" not in yml_random) or ("window" not in yml_random):
                with open(self.yml_log, "w") as ymlseqfile:
                    yaml.dump(self.data_yml_log, ymlseqfile, default_flow_style=False)

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


class AwarenessSession(PylinkEyetrackerSession):
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
        self.task = task
        self.data_yml_log = {}

        # Create log folder if it does not exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # set realtime mode for higher timing precision
        pylink.beginRealTimeMode(100)

        self._create_locations()
        self._create_yaml_log()
        self._create_fixation()
        self._create_stimuli()
        self.create_sequences()
        self.create_trials()
        self.save_yaml_log()

        print("--------------------------------")
        print(
            "    /\\_/\\           ___\n   = o_o =_______    \\ \\ \n    __^      __(  \.__) )\n(@)<_____>__(_____)____/"
        )
        print("Author: @Ningkai Wang")
        print("--------------------------------")

    def _create_locations(self):
        self.angles = [
            int(i) for i in np.linspace(45, 360 + 45, 4, endpoint=False)
        ] % np.array([360])

    def _create_fixation(self):
        self.fixbullseye = FixationBullsEye(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("distance_from_center"),
            color=(0, 0, 0, 1),
            pos=[0, self.roll_dist],
            **{
                "lineWidth": monitorunittools.deg2pix(
                    self.settings["stimuli"].get("outer_fix_linewidth"),
                    self.win.monitor,
                )
            },
        )

        self.fixation_dot = FixationDot(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("fixation_size_deg"),
            pos=[0, self.roll_dist],
            dotcolor=-1,
            linecolor=self.settings["window"].get("color"),
            cross_lindwidth=monitorunittools.deg2pix(
                self.settings["stimuli"].get("fixation_size_deg") / 5, self.win.monitor
            ),
        )

    def _create_stimuli(self):
        self.placeholders = {}
        for angle in self.angles:
            self.placeholders[(angle)] = PlaceHolder(
                win=self.win,
                circle_radius=self.settings["stimuli"].get("stim_size_deg"),
                color=0.5,
                ecc=self.settings["stimuli"].get("distance_from_center"),
                roll_dist=self.roll_dist,
                angle=angle,
                linewidth=monitorunittools.deg2pix(
                    self.settings["stimuli"].get("outer_fix_linewidth"),
                    self.win.monitor,
                ),
            )
            self.placeholders[(angle)].draw()

        self.highlighters = {}
        for angle in self.angles:
            self.highlighters[angle] = Highlighter(
                win=self.win,
                circle_radius=self.settings["stimuli"].get("stim_size_deg"),
                linecolor=1,
                ecc=self.settings["stimuli"].get("distance_from_center"),
                roll_dist=self.roll_dist,
                angle=angle,
                fillcolor=self.settings["window"].get("color"),
                linewidth=monitorunittools.deg2pix(
                    self.settings["stimuli"].get("outer_fix_linewidth") * 2,
                    self.win.monitor,
                ),
            )
            self.highlighters[angle].draw()

        self.fsmask = Circle(
            win=self.win,
            units="deg",
            radius=50,
            pos=[0, 0],
            edges=360,
            fillColor=self.settings["window"].get("color"),
            lineColor=self.settings["window"].get("color"),
        )
        self.fsmask.draw()
        self.win.flip()

    def create_sequences(self):
        self.seq_awareness_check = list(itertools.combinations(self.angles, 2))
        np.random.shuffle(self.seq_awareness_check)
        self.seq_awareness_check_single = [(self.angles[i]) for i in range(4)]
        np.random.shuffle(self.seq_awareness_check_single)
        [
            self.seq_awareness_check.append([loc])
            for loc in self.seq_awareness_check_single
        ]

    def create_trials(self):
        self.trial_counter = 0
        self.trials = []
        self.trials.append(
            InstructionTrial_awareness(
                session=self,
                trial_nr=self.trial_counter,
                phase_durations=[np.inf],
                keys=self.settings["various"].get("buttons_test"),
                txt=self.settings["stimuli"].get("awareness_instruction_text"),
                txt_height=self.settings["various"].get("text_height"),
                txt_width=self.settings["various"].get("text_width"),
                txt_position_x=self.settings["various"].get("text_position_x"),
                txt_position_y=self.settings["various"].get("text_position_y")
                + self.roll_dist
                - 1,
                image=os.path.join(
                    parent_dir,
                    "stimuli",
                    self.settings["stimuli"].get("awareness_instruction_image"),
                ),
                draw_each_frame=False,
            )
        )

        for highlighted in self.seq_awareness_check:
            parameters = {
                "highlighted": list(highlighted),
            }
            self.trials.append(
                AwarenessCheckTrial(
                    session=self,
                    trial_nr=self.trial_counter,
                    phase_durations=[np.inf],
                    phase_names=["awareness_check"],
                    parameters=parameters,
                    keys=self.settings["various"].get("buttons_test"),
                    draw_each_frame=False,
                )
            )
            self.trial_counter += 1
        pass

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
                    yml_random = yaml.safe_load(ymlseqfile)
                except yaml.YAMLError as exc:
                    print(exc)
            logging.warn("Subject yaml log file loaded:" + str(yml_random))
            self.data_yml_log = yml_random
            if yml_random.get("window") is None:
                create_new_roll_dist = True
            else:
                self.roll_dist = yml_random.get("window").get("roll_dist")
                if self.roll_dist is None:
                    create_new_roll_dist = True
                else:
                    create_new_roll_dist = False
        else:
            create_new_roll_dist = True

        if create_new_roll_dist:
            self.roll_dist = 0
            self.data_yml_log["window"] = {
                "roll_dist": self.roll_dist,
            }

    def save_yaml_log(self):
        logging.warn("running save_yaml_log")
        if not os.path.isfile(self.yml_log):
            logging.warn("yml log file not found, creating a new one")
            with open(self.yml_log, "w") as ymlseqfile:
                yaml.dump(self.data_yml_log, ymlseqfile, default_flow_style=False)
        else:
            with open(self.yml_log, "r") as ymlseqfile:
                try:
                    yml_random = yaml.safe_load(ymlseqfile)
                except yaml.YAMLError as exc:
                    print(exc)

            if "window" not in yml_random:
                logging.warn("window not in ymlseqfile")
                logging.warn(self.data_yml_log)
                with open(self.yml_log, "w") as ymlseqfile:
                    yaml.dump(self.data_yml_log, ymlseqfile, default_flow_style=False)
            elif yml_random.get("window").get("roll_dist") != self.roll_dist:
                logging.warn("roll_dist changed")
                with open(self.yml_log, "w") as ymlseqfile:
                    yaml.dump(self.data_yml_log, ymlseqfile, default_flow_style=False)
            else:
                logging.warn("roll_dist not changed")
                pass

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
