#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from stimuli import FixationBullsEye, FixationCue, PlaceHolderCircles
from trial import (
    SequenceTrial,
    InstructionTrial,
    DummyWaiterTrial,
    OutroTrial,
    LocalizerTrial,
    FeedbackTrial,
)
from psychopy import sound

rng = random.SystemRandom()


class PredSession(PylinkEyetrackerSession):
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
        self.run_nr = run_nr
        self.data_yml_random_log = {}

        # Create log folder if it does not exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # stimulus materials
        self.stim_file_path = os.path.join(
            os.path.split(__file__)[0],
            "stimuli",
            self.settings["stimuli"].get("stim_h5file"),
        )
        self.cue_dir_path = os.path.join(
            os.path.split(__file__)[0], "stimuli", "cue_object"
        )
        if not os.path.isfile(self.stim_file_path):
            logging.warn(
                f"Downloading stimulus file from figshare to {self.stim_file_path}"
            )
            urllib.request.urlretrieve(
                self.settings["stimuli"].get("stim_url"), self.stim_file_path
            )
        try:
            self.port = parallel.ParallelPort(address=0x0378)
            self.port.setData(0)
            self.parallel_triggering = True
        except:
            logging.warn(f"Attempted import of Parallel Port failed")
            self.parallel_triggering = False

        # set realtime mode for higher timing precision
        pylink.beginRealTimeMode(100)

        self._create_text_loading()
        self._create_parameters()
        self._create_stimuli()

        self.create_sequences()
        self.save_yaml_random_log()
        self.create_trials()

        for i, seq in enumerate(range(len(self.seq_seq))):
            print(
                seq,
                self.seq_cue_object[i],
                "-->",
                self.seq_subcate[i][0],
                self.seq_cate_ind[i][0],
                self.seq_location[i][0],
                "-->",
                self.seq_subcate[i][1],
                self.seq_cate_ind[i][1],
                self.seq_location[i][1],
                "-->",
                self.seq_subcate[i][2],
                self.seq_cate_ind[i][2],
                self.seq_location[i][2],
                "-->",
                self.seq_subcate[i][3],
                self.seq_cate_ind[i][3],
                self.seq_location[i][3],
                "-------",
                "Type_test:",
                self.type_test[i],
                "-------",
                "Signal_trial:",
                self.seq_AT[i].count(True),
            )

        print("--------------------------------")
        print(
            "    /\\_/\\           ___\n   = o_o =_______    \\ \\ \n    __^      __(  \.__) )\n(@)<_____>__(_____)____/"
        )
        print("Author: @Ningkai Wang")
        print("--------------------------------")

    def create_sequences(self):
        """
        Creates all sequences used in the experiment.
        If there exists a sequence log file, it will be used to recreate the sequences.
        If the sequence log file does not exist, the sequences will be created from scratch.

        variables:
        self.seq_seq:
            - used to indicate the [cue] -> [face->scene->face->scene] sequence.
        self.seq_orig_loc:
            - indicate the original location of the sequence in each trial.
            - struct: [[0, 1, 2, 3], [1, 2, 3, 0]]
        self.seq_location:
            - indicate the location of each picture in the sequence in each trial.
        self.seq_cate:
            - indicate value=face/scene in each trial.
        self.seq_subcate:
            - indicate value=male/female/indoor/outdoor in each trial.
            - struct: ['face', 'scene', 'scene', 'face'], ['scene', 'face', 'scene', 'face']
        self.seq_cate_ind:
            - indicate vaule= in each trial.
            - struct: [[0,1,1,0],[1,1,0,1]]
        """

        # create ses sequence
        if self.ses_nr == 0:
            self.seq_orig_loc = np.tile(
                self.sequence_clockwise,
                [
                    self.settings["design"].get("nr_per_full_sequence")
                    * self.nr_diff_sequences,
                    1,
                ],
            )
            self.seq_seq = np.tile(
                [*range(self.nr_diff_sequences)],
                self.settings["design"].get("nr_per_full_sequence"),
            )
            self.nr_per_sequence = self.settings["design"].get("nr_per_full_sequence")
        elif self.ses_nr == 1:
            self.seq_orig_loc = np.tile(
                self.sequence_clockwise,
                [
                    self.settings["design"].get("nr_per_partial_sequence")
                    * self.nr_diff_sequences,
                    1,
                ],
            )
            self.seq_seq = np.tile(
                [*range(self.nr_diff_sequences)],
                self.settings["design"].get("nr_per_partial_sequence"),
            )
            self.nr_per_sequence = self.settings["design"].get(
                "nr_per_partial_sequence"
            )
        elif self.ses_nr == 2:
            self.seq_orig_loc = np.tile(
                self.sequence_clockwise,
                [
                    self.settings["design"].get("nr_per_violate_sequence")
                    * self.nr_diff_sequences,
                    1,
                ],
            )
            self.seq_seq = np.tile(
                [*range(self.nr_diff_sequences)],
                self.settings["design"].get("nr_per_violate_sequence"),
            )
            self.nr_per_sequence = self.settings["design"].get(
                "nr_per_violate_sequence"
            )

        else:
            raise ValueError(
                "ses_nr must be 0 (training full sequence), 1 (partial sequence) or 2 (violate sequence)"
            )

        # load shuflled sequence
        """Keep it here for test"""
        if self.nr_per_sequence == 20 and self.nr_diff_sequences == 6:
            with h5py.File(
                os.path.join(
                    os.path.split(__file__)[0],
                    "stimuli",
                    self.settings["stimuli"].get("sim_seq_h5file"),
                ),
                "r",
            ) as h5simfile:
                seq_1000 = list(h5simfile["sequence"])
                sim_seq_ind = rng.randint(0, len(seq_1000) - 1)
                self.seq_seq = seq_1000[sim_seq_ind]
                ITI_expon_1000 = list(h5simfile.get("ITI_expon"))
                self.ITI_expon = ITI_expon_1000[sim_seq_ind]
            print(
                "nr per sequence: ",
                self.nr_per_sequence,
                "will use simulated sequences.",
            )
        else:
            rng.shuffle(self.seq_seq)
            # self.ITI_expon = []
            # for _ in range(len(self.seq_seq)):
            #     self.ITI_expon.append(rng.sample(self.settings['stimuli'].get('ITI_time'), 1)[0])
            self.ITI_expon = expon.rvs(
                scale=self.settings["stimuli"].get("ITI_time").get("scale"),
                loc=self.settings["stimuli"].get("ITI_time").get("loc"),
                size=len(self.seq_seq),
            )
            print(
                "nr per sequence: ", self.nr_per_sequence, "will shuffle it randomly."
            )

        # create category sequence, the seq_cate variable (value=male/female/indoor/outdoor) will be used in trials
        self.seq_cue_object = []  # list of cue colors for each sequence
        self.seq_cate = []
        self.seq_subcate = []
        self.seq_location = []
        self.seq_cate_ind = []

        ## create category and location sequences
        random_location = self.settings["design"].get("location_name").copy()
        shuffle_exem = list(
            zip(
                self.seq_cate_exem.copy(),
                self.seq_subcate_exem.copy(),
                self.cate_ind_exem["img_ind"].copy(),
            )
        )

        for seq in self.seq_seq:
            self.seq_cue_object.append(self.cue_object[seq])
            if seq in [*range(self.nr_stru_seqs)]:
                self.seq_cate.append(self.seq_cate_stru)
                self.seq_location.append(
                    [
                        self.settings["design"].get("location_name")[x]
                        for x in self.sequence_location[seq]
                    ]
                )
                self.seq_subcate.append(self.seq_subcate_stru.copy())
                self.seq_cate_ind.append(self.cate_ind_stru["img_ind"].copy())
            elif seq in list(
                set([*range(self.nr_exem_seqs + self.nr_stru_seqs)])
                - set([*range(self.nr_stru_seqs)])
            ):
                random_location = rng.sample(
                    self.sequence_location_perm[self.nr_stru_seqs : :], 1
                )[0]
                rng.shuffle(shuffle_exem)
                random_cate_exem, random_subcate_exem, random_cateind_exem = zip(
                    *shuffle_exem
                )
                self.seq_location.append(
                    [
                        self.settings["design"].get("location_name")[x]
                        for x in random_location
                    ]
                )
                self.seq_cate.append(list(random_cate_exem).copy())
                self.seq_subcate.append(list(random_subcate_exem).copy())
                self.seq_cate_ind.append(list(random_cateind_exem).copy())
            elif seq in list(
                set([*range(self.nr_diff_sequences)])
                - set([*range(self.nr_exem_seqs + self.nr_stru_seqs)])
            ):
                ncate = rng.sample(self.sequence_category_perm, 1)[0]
                self.seq_cate.append(
                    [
                        self.settings["design"].get("category_key")[ncate[x]]
                        for x in [*range(len(self.sequence_clockwise))]
                    ]
                )
                random_location = rng.sample(
                    self.sequence_location_perm[self.nr_stru_seqs : :], 1
                )[0]
                self.seq_location.append(
                    [
                        self.settings["design"].get("location_name")[x]
                        for x in random_location
                    ]
                )
                self.seq_subcate.append(
                    [
                        self.settings["design"]
                        .get("category")
                        .get(x)[
                            rng.sample(
                                range(
                                    len(self.settings["design"].get("category").get(x))
                                ),
                                1,
                            )[0]
                        ]
                        for x in self.seq_cate[-1]
                    ]
                )
                self.seq_cate_ind.append(
                    [
                        rng.sample(self.img_pool_blk[key], 1)[0]
                        for key in self.seq_subcate[-1]
                    ]
                )
            else:
                raise ValueError("seq must be in range of nr_diff_sequences")

        # create test_sequences
        self._create_test_sequence()

        # create attention tracking task
        self._create_AT_trials()
        ########################################

        # create sequence response
        self._create_sequence_response()

        # create instruction text
        self.instruction_text = self.settings["stimuli"].get("instruction_text")
        if self.ses_nr == 0 or self.ses_nr == 2 or self.ses_nr == 2:
            self.instruction_text = (
                eval(f"f'{self.instruction_text}'")
                + f"\n\n When you see greyscale pictures, please press Spacebar!"
            )

    def _create_stimuli(self):
        """Creates all stimuli used in the experiment."""

        # Create picture locations
        self._create_location()

        # Create masks
        self._create_mask()

        # create fixation cross
        self._create_fixation()

        # Create the face & scene stimuli
        self.images = {}
        self.image_stims = {}
        for cate in self.stimuli_number:
            self.images[cate] = {}
            for loc in self.stim_location:
                self.images[cate][loc] = self.images_all[cate].copy()
        for img_type in ["greyscaled", "colored"]:
            self.image_stims[img_type] = {}
            for cate in self.stimuli_number:
                self.image_stims[img_type][cate] = {}
                for loc in self.stim_location:
                    self.image_stims[img_type][cate][loc] = []
                    for ind_img, img in enumerate(self.images[cate][loc]):
                        if img_type == "greyscaled":
                            # img = np.flipud(img)
                            img = Image.fromarray(np.uint8(img * 255)).convert("L")
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)
                            texRes = img.width
                            # texRes = img.shape[1]
                        elif img_type == "colored":
                            texRes = img.shape[1]
                        self.image_stims[img_type][cate][loc].append(
                            ImageStim(
                                win=self.win,
                                image=img,
                                mask="circle",
                                pos=self.stim_location[loc],
                                units="deg",
                                name=cate + "_" + str(ind_img) + "_" + loc,
                                texRes=texRes,
                                colorSpace="rgb",
                                size=self.settings["stimuli"].get("stim_size_deg"),
                                interpolate=True,
                            )
                        )
                    for ibs in self.image_stims[img_type][cate][loc]:
                        ibs.draw()  # draw all the bg stimuli once, before they are used in the trials

        for loc in self.stim_location:
            self.intromask[loc].draw()

        self.beep = sound.Sound(
            os.path.join(os.path.split(__file__)[0], "stimuli", "beep.wav")
        )
        self.win.flip()
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

        # Create phase_durations
        self.phase_durations = []
        phase_durations = np.tile(
            [
                self.settings["stimuli"].get("stim_refresh_time"),
                self.settings["stimuli"].get("seq_overlap_time"),
            ],
            len(self.settings["design"].get("sequence_clockwise")),
        ).tolist()
        phase_durations.insert(0, self.settings["stimuli"].get("fixdot_refresh_time"))
        for seq_ind in self.seq_seq:
            x = phase_durations.copy()
            x.append(self.ITI_expon[seq_ind])
            self.phase_durations.append(
                x
            )  # add the ITI time to the end of the sequence

        # Create trials
        self.res_feedback = [0, 0]
        for _ in self.seq_orig_loc:
            parameters   = {'trial_ind': int(self.trial_counter-2),
                            'cue_object': self.cue_object[self.seq_seq[self.trial_counter-2]],
                            'location': self.seq_location[self.trial_counter-2],
                            'category': self.seq_cate[self.trial_counter-2],
                            'subcategory': self.seq_subcate[self.trial_counter-2],
                            'img_ind': self.seq_cate_ind[self.trial_counter-2],
                            'ind_test': self.ind_test[self.trial_counter-2],
                            'correct_kb': self.seq_response[self.trial_counter-2],
                            'AT_task': self.seq_AT[self.trial_counter-2],
                            'type_test': self.type_test[self.trial_counter-2],
                            }

            self.res_feedback.append(0)
            self.trials.append(
                SequenceTrial(
                    session=self,
                    trial_nr=self.trial_counter,
                    phase_durations=self.phase_durations[self.trial_counter - 2],
                    phase_names=[
                        "cue_0",
                        "stim_0",
                        "interval_0",
                        "stim_1",
                        "interval_1",
                        "stim_2",
                        "interval_2",
                        "stim_3",
                        "interval_3",
                        "ITI",
                    ],
                    parameters=parameters,
                    timing="seconds",
                    verbose=True,
                    draw_each_frame=False,
                )
            )
            self.trial_counter += 1

        self.trials.append(FeedbackTrial(session=self, trial_nr=self.trial_counter))

    def _create_test_sequence(self):
        """Creates a partial sequence (ses_nr == 1) or a violate sequence sequence (ses_nr == 2)"""
        self.ind_test = []
        self.type_test = []
        if self.ses_nr == 0:
            for i in range(len(self.seq_cate)):
                self.ind_test.append(None)
                self.type_test.append(None)
        elif self.ses_nr == 1:
            self.partial_seq_trial = [rng.random() for i in range(len(self.seq_seq))]
            for i in range(len(self.seq_cate)):
                # determine when to start a partial part
                # self.ind_test.append(rng.sample(self.settings['design'].get('sequence_clockwise'), 1)[0])
                # determine wether it is a partial part or not
                if self.partial_seq_trial[i] >= self.settings["design"].get(
                    "ratio_partial_sequences"
                ):
                    self.ind_test.append(3)
                    self.type_test.append(None)
                else:
                    self.ind_test.append(-1)
                    self.type_test.append("partial")

                # self.ind_test.append(rng.sample([0], 1)[0])
        elif self.ses_nr == 2:
            # In this part we need to exchange two pictures or two locations
            nr_per_violate_sequence = self.settings["design"].get(
                "nr_per_violate_sequence"
            )
            nr_violate = nr_per_violate_sequence * self.settings["design"].get(
                "ratio_violate_sequences"
            )
            ri = [*range(math.ceil(nr_per_violate_sequence))]
            rng.shuffle(ri)
            k = 0

            for i, seq in enumerate(self.seq_seq):
                if seq in [*range(self.nr_stru_seqs)]:
                    if ri[k] < nr_violate:
                        # Exchange pictures
                        """
                        Or borrow pics from the exemplar sequence?
                        """
                        if ri[k] < nr_violate / 3:
                            c = self.seq_cate[i][3]
                            sc = self.seq_subcate[i][3]
                            ind = self.seq_cate_ind[i][3]
                            self.seq_cate[i][3] = self.seq_cate[i][2]
                            self.seq_subcate[i][3] = self.seq_subcate[i][2]
                            self.seq_cate_ind[i][3] = self.seq_cate_ind[i][2]
                            self.seq_cate[i][2] = c
                            self.seq_subcate[i][2] = sc
                            self.seq_cate_ind[i][2] = ind
                            self.type_test.append("violate_pic")
                        # Exchange locations
                        elif ri[k] >= nr_violate / 3 and ri[k] < nr_violate * 2 / 3:
                            l = self.seq_location[i][3]
                            self.seq_location[i][3] = self.seq_location[i][2]
                            self.seq_location[i][2] = l
                            self.type_test.append('violate_loc')
                            """
                            If this part is changed in the future, make sure to add the variable to the parameters when creating the trial
                            """
                        # Exchange both
                        elif ri[k] >= nr_violate * 2 / 3 and ri[k] < nr_violate:
                            c = self.seq_cate[i][3]
                            sc = self.seq_subcate[i][3]
                            ind = self.seq_cate_ind[i][3]
                            l = self.seq_location[i][3]
                            self.seq_cate[i][3] = self.seq_cate[i][2]
                            self.seq_subcate[i][3] = self.seq_subcate[i][2]
                            self.seq_cate_ind[i][3] = self.seq_cate_ind[i][2]
                            self.seq_location[i][3] = self.seq_location[i][2]
                            self.seq_cate[i][2] = c
                            self.seq_subcate[i][2] = sc
                            self.seq_cate_ind[i][2] = ind
                            self.seq_location[i][2] = l
                            self.type_test.append("violate_both")
                        self.ind_test.append(2)
                    else:
                        self.ind_test.append(None)
                        self.type_test.append(None)
                    k += 1
                else:
                    self.ind_test.append(None)
                    self.type_test.append(None)
        else:
            raise ValueError(
                "ses_nr must be 0 (training full sequence), 1 (partial sequence), or 2 (violate sequence)"
            )

    def _shuffle_buttons(self):
        """Shuffles the button order"""
        rng.shuffle(self.buttons)
        for i in range(len(self.buttons)):
            rng.shuffle(self.buttons[i])

    def _create_exemplar_sequences(self):
        """Creates the exemplar sequences"""

    def _create_AT_trials(self):
        """
        Creates the attention tracking trials
        -------------------------------------
        This method will first caculate the number of trials that will be used for the attention tracking,
        and determine the AT image, and then create the trials.
        """

        """









        
        UNDER CONSTRUCTION//
        Maybe increase the rate of the AT trials in the violation session
        









        """
        self.nr_ATtrials = int(
            math.ceil(self.settings["design"].get("rate_ATtrial") * len(self.seq_seq))
        )
        if self.ses_nr == 2:
            self.nr_ATtrials = self.nr_ATtrials * 2
        self.seq_AT = [
            [False] * len(self.sequence_clockwise) for _ in range(len(self.seq_seq))
        ]

        for seq in range(self.nr_diff_sequences):
            AT_trials_inds = np.where(np.array(self.seq_seq) == seq)[0]
            AT_trials_ind = rng.sample(
                list(AT_trials_inds),
                math.ceil(self.nr_ATtrials / self.nr_diff_sequences),
            )  # get the indices of the AT trials
            for i in AT_trials_ind:
                k = rng.sample(range(len(self.sequence_clockwise)), 1)[0]
                self.seq_AT[i][k] = True

    def _create_sequence_response(self):
        """Press space to response the specific picture"""
        self.seq_response = [
            [None] * len(self.sequence_clockwise) for _ in [*range(len(self.seq_seq))]
        ]
        for i in range(len(self.seq_AT)):
            for j in range(len(self.seq_AT[i])):
                if self.seq_AT[i][j]:
                    self.seq_response[i][j] = self.buttons[0][0]

        # No response for the empty part of trial
        if self.ses_nr == 1:
            for i in range(len(self.seq_response)):
                for j in range(len(self.seq_response[i])):
                    if j >= self.ind_test[i]:
                        self.seq_response[i][j] = None

    def _create_sequence_response_2(self):
        """Creates the response sequence (for d/f/j/k responses)"""
        self.seq_response = self.seq_subcate.copy()
        self.buttons_map = {}
        for i in [*range(len(self.settings["design"].get("category_key")))]:
            for j in [
                *range(
                    len(
                        self.settings["design"].get("category")[
                            self.settings["design"].get("category_key")[i]
                        ]
                    )
                )
            ]:
                self.buttons_map[
                    self.settings["design"].get("category")[
                        self.settings["design"].get("category_key")[i]
                    ][j]
                ] = self.buttons[i][j]

        for i in [*range(len(self.seq_subcate))]:
            self.seq_response[i] = [self.buttons_map[x] for x in self.seq_subcate[i]]
        # No response for the empty part of trial
        if self.ses_nr == 1:
            for i in [*range(len(self.seq_subcate))]:
                for j in [*range(len(self.seq_subcate[i]))]:
                    if j >= self.ind_test[i]:
                        self.seq_response[i][j] = None

    def _randomize_block_origin(self):
        for i in [*range(len(self.seq_orig_loc))]:
            self.seq_orig_loc[i] = [
                (x + rng.sample(self.seq_orig_loc[i].tolist(), 1)[0])
                % len(self.seq_orig_loc[i])
                for x in self.seq_orig_loc[i]
            ]

        rng.shuffle(self.seq_orig_loc)

    def _create_location(self):
        """Creates the location of the sequence"""
        self.distance_from_center = self.settings["stimuli"].get(
            "distance_from_center"
        ) * math.sqrt(1 / 2)
        self.stim_location = self.settings["design"].get("stim_location")
        for key in self.stim_location:
            self.stim_location[key] = [
                x * self.distance_from_center for x in self.stim_location[key]
            ]

    def _create_fixation(self):
        self.fixation_cue = {}
        self.fixation_w = FixationBullsEye(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("fixation_size_pixels"),
            color=(0.5, 0.5, 0.5, 1),
            **{"lineWidth": self.settings["stimuli"].get("outer_fix_linewidth")},
        )

        for cueobject in self.cue_object:
            img = os.path.join(self.cue_dir_path, cueobject)
            self.fixation_cue[cueobject] = ImageStim(
                win=self.win,
                image=img,
                mask="circle",
                pos=[0, 0],
                units="deg",
                name=cueobject,
                texRes=256,
                colorSpace="rgb",
                size=self.settings["stimuli"].get("cue_size_deg"),
                interpolate=True,
            )
            self.fixation_cue[cueobject].draw()

        self.fixation_dot = FixationCue(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("cue_size_pixels"),
            color=-1,
            **{"lineWidth": self.settings["stimuli"].get("outer_fix_linewidth")},
        )

        self.filled_circles = PlaceHolderCircles(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("stim_size_deg"),
            color=-0.5,
            ecc=self.settings["stimuli"].get("distance_from_center"),
            linewidth=self.settings["stimuli"].get("stim_circle_linewidth"),
        )
        self.filled_circles.draw()
        self.empty_circles = PlaceHolderCircles(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("stim_size_deg"),
            color=-0.5,
            ecc=self.settings["stimuli"].get("distance_from_center"),
            linewidth=self.settings["stimuli"].get("stim_circle_linewidth"),
            fill=False,
        )
        self.empty_circles.draw()

        self.fixation_dot.draw()
        self.cuemask.draw()
        for loc in self.stim_location:
            self.intromask[loc].draw()

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

    def _create_mask(self):
        self.cuemask = GratingStim(
            self.win,
            tex=np.ones((4, 4)),
            pos=[0, 0],
            units="deg",
            contrast=1,
            color=(0.0, 0.0, 0.0),
            name="cue_mask",
            size=self.settings["stimuli"].get("cue_size_deg"),
        )
        self.intromask = {}
        for loc in self.stim_location:
            self.intromask[loc] = GratingStim(
                self.win,
                tex=np.ones((4, 4)),
                pos=self.stim_location[loc],
                units="deg",
                contrast=1,
                color=(0.0, 0.0, 0.0),
                name="intromask" + "_" + loc,
                size=self.settings["stimuli"].get("stim_size_deg"),
            )

    def _create_parameters(self):
        # load image
        self.images_all = {}
        self.stimuli_number = {}
        with h5py.File(
            os.path.join(
                os.path.split(__file__)[0],
                "stimuli",
                self.settings["stimuli"].get("stim_h5file"),
            ),
            "r",
        ) as h5stimfile:
            for key in h5stimfile:
                self.stimuli_number[key] = h5stimfile.get(key).shape[0]
            # for key in h5stimfile:
            # self.images_all[key] = np.array(h5stimfile.get(key)) / 256

        # every n block, use the new sequences and the new stimuli
        log_run = self.run_nr - self.run_nr % self.settings["design"].get(
            "nr_renew_blocks"
        )
        self.yml_random_log = os.path.join(
            self.output_dir,
            f"sub-{str(self.subject).zfill(2)}_ses-{str(self.ses_nr).zfill(2)}_task-pred_run-{str(log_run).zfill(2)}"
            + "_random_log.yml",
        )
        self.cue_object = [
            os.path.basename(x) for x in glob.glob(self.cue_dir_path + "/*png")
        ]
        self.nr_stru_seqs = self.settings["design"].get("nr_stru_seqs")
        self.nr_exem_seqs = self.settings["design"].get("nr_exem_seqs")
        self.nr_rand_seqs = self.settings["design"].get("nr_rand_seqs")
        self.nr_diff_sequences = (
            self.nr_stru_seqs + self.nr_exem_seqs + self.nr_rand_seqs
        )
        # determine if there is a sequence log file. If so, load it.
        if os.path.isfile(self.yml_random_log):
            with open(self.yml_random_log, "r") as ymlseqfile:
                try:
                    yml_random = yaml.safe_load(ymlseqfile)
                except yaml.YAMLError as exc:
                    print(exc)

            self.cue_object = yml_random.get("sequence").get("cue_object")
            self.sequence_clockwise = yml_random.get("sequence").get(
                "sequence_clockwise"
            )
            self.sequence_location = yml_random.get("sequence").get("sequence_location")
            self.sequence_category = yml_random.get("sequence").get("sequence_category")
            self.sequence_category_perm = yml_random.get("sequence").get(
                "sequence_category_perm"
            )
            self.buttons = yml_random.get("sequence").get("buttons")
            self.nr_images_per_category = yml_random.get("sequence").get(
                "nr_images_per_category"
            )
            self.sequence_location_perm = yml_random.get("sequence").get(
                "sequence_location_perm"
            )
            self.seq_cate_stru = yml_random.get("info_exem").get("seq_cate_stru")
            self.seq_cate_exem = yml_random.get("info_exem").get("seq_cate_exem")
            self.seq_subcate_stru = yml_random.get("info_exem").get("seq_subcate_stru")
            self.seq_subcate_exem = yml_random.get("info_exem").get("seq_subcate_exem")
            self.cate_ind_stru = yml_random.get("info_exem").get("cate_ind_stru")
            self.cate_ind_exem = yml_random.get("info_exem").get("cate_ind_exem")
            self.img_pool_blk = {}
            for key in self.stimuli_number:
                self.img_pool_blk[key] = [*range(self.nr_images_per_category)]
                if key in self.seq_subcate_stru:
                    x = np.array(self.seq_subcate_stru)
                    y = np.where(x == key)[0]
                    for i in y:
                        self.img_pool_blk[key] = list(
                            set(self.img_pool_blk[key])
                            - set([self.cate_ind_stru["img_ind"][i]])
                        )
                if key in self.seq_subcate_exem:
                    x = np.array(self.seq_subcate_exem)
                    y = np.where(x == key)[0]
                    for i in y:
                        self.img_pool_blk[key] = list(
                            set(self.img_pool_blk[key])
                            - set([self.cate_ind_exem["img_ind"][i]])
                        )

        # if there is no sequence log file, create the sequences from scratch
        else:
            self.img_pool = {}
            for key in self.stimuli_number:
                self.img_pool[key] = [*range(self.stimuli_number[key])]
            for run in range(self.run_nr):
                self.yml_random_log_prev_run = os.path.join(
                    self.output_dir,
                    f"sub-{str(self.subject).zfill(2)}_ses-{str(self.ses_nr).zfill(2)}_task-pred_run-{str(run).zfill(2)}"
                    + "_random_log.yml",
                )
                if os.path.isfile(self.yml_random_log_prev_run):
                    # remove the face/scene images that have been used in the previous run
                    with open(self.yml_random_log_prev_run, "r") as ymlseqfile:
                        try:
                            yml_random_prev_run = yaml.safe_load(ymlseqfile)
                        except yaml.YAMLError as exc:
                            print(exc)
                    cue_object_prev_run = yml_random_prev_run.get("sequence").get(
                        "cue_object"
                    )
                    self.cue_object = list(
                        set(self.cue_object) - set(cue_object_prev_run)
                    )
                    image_index_prev_run = {}
                    for key in self.stimuli_number:
                        image_index_prev_run[key] = yml_random_prev_run.get(
                            "image_index"
                        ).get(key)
                        self.img_pool[key] = list(
                            set(self.img_pool[key]) - set(image_index_prev_run[key])
                        )

            rng.shuffle(self.cue_object)
            self.cue_object = self.cue_object[: self.nr_diff_sequences]

            self.sequence_clockwise = self.settings["design"].get("sequence_clockwise")

            self.sequence_location_perm = list(
                itertools.permutations(
                    self.settings["design"].get("sequence_clockwise")
                )
            )

            self.sequence_location_perm = [list(x) for x in self.sequence_location_perm]
            rng.shuffle(self.sequence_location_perm)

            self.sequence_location = self.sequence_location_perm[0 : self.nr_stru_seqs]

            self.sequence_category_perm = self.settings["design"].get(
                "sequence_category_perm"
            )

            self.nr_images_per_category = self.settings["stimuli"].get(
                "nr_images_per_category"
            )

            rng.shuffle(self.sequence_location)
            rng.shuffle(self.sequence_category_perm)

            self.img_pool_blk = {}
            for key in self.stimuli_number:
                self.img_pool_blk[key] = [*range(self.nr_images_per_category)]

            self.sequence_category = self.sequence_category_perm[
                0 : self.nr_stru_seqs + self.nr_exem_seqs
            ]

            self.buttons = self.settings["various"].get("buttons")
            self._shuffle_buttons()

            self.data_yml_random_log["sequence"] = {
                "cue_object": self.cue_object,
                "sequence_clockwise": self.sequence_clockwise,
                "sequence_location": deepcopy(list(self.sequence_location)),
                "sequence_category": deepcopy(list(self.sequence_category)),
                "sequence_category_perm": deepcopy(list(self.sequence_category_perm)),
                "buttons": self.buttons,
                "nr_images_per_category": self.nr_images_per_category,
                "sequence_location_perm": self.sequence_location_perm,
            }

        # Choose the images to use for the block
        if os.path.isfile(self.yml_random_log):
            with open(self.yml_random_log, "r") as ymlseqfile:
                try:
                    yml_random = yaml.safe_load(ymlseqfile)
                    self.image_index = {}
                    for key in self.stimuli_number:
                        self.image_index[key] = yml_random.get("image_index").get(key)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            self.image_index = {}
            self.data_yml_random_log["image_index"] = {}
            for key in self.stimuli_number:
                self.image_index[key] = rng.sample(
                    self.img_pool[key], self.nr_images_per_category
                )
                self.data_yml_random_log["image_index"][key] = self.image_index[key]

            self.seq_cate_stru = [
                self.settings["design"].get("category_key")[x]
                for x in self.sequence_category[0]
            ]  # category of the structure sequences, nubmer of sequence set as 1
            self.seq_cate_exem = [
                self.settings["design"].get("category_key")[x]
                for x in self.sequence_category[1]
            ]  # category of the structure sequences, nubmer of sequence set as 1
            self.seq_subcate_stru = [
                self.settings["design"]
                .get("category")
                .get(x)[
                    rng.sample(
                        range(len(self.settings["design"].get("category").get(x))), 1
                    )[0]
                ]
                for x in self.seq_cate_stru
            ]
            self.seq_subcate_exem = [
                self.settings["design"]
                .get("category")
                .get(x)[
                    rng.sample(
                        range(len(self.settings["design"].get("category").get(x))), 1
                    )[0]
                ]
                for x in self.seq_cate_exem
            ]
            self.data_yml_random_log["info_exem"] = {}
            self.data_yml_random_log["info_exem"]["seq_cate_stru"] = self.seq_cate_stru
            self.data_yml_random_log["info_exem"]["seq_cate_exem"] = self.seq_cate_exem
            self.data_yml_random_log["info_exem"][
                "seq_subcate_stru"
            ] = self.seq_subcate_stru
            self.data_yml_random_log["info_exem"][
                "seq_subcate_exem"
            ] = self.seq_subcate_exem

            self.cate_ind_stru, self.cate_ind_exem = {}, {}
            self.cate_ind_stru["img_ind"], self.cate_ind_exem["img_ind"] = [], []
            for key in self.stimuli_number:
                isc = rng.sample(range(len(self.image_index[key])), 1)[0]
                self.img_pool_blk[key] = list(set(self.img_pool_blk[key]) - set([isc]))
                self.cate_ind_stru["img_ind"].append(isc)
                iec = rng.sample(range(len(self.image_index[key])), 1)[0]
                self.img_pool_blk[key] = list(set(self.img_pool_blk[key]) - set([iec]))
                self.cate_ind_exem["img_ind"].append(iec)

            for key in self.stimuli_number:
                if key in self.seq_subcate_stru:
                    x = np.array(self.seq_subcate_stru)
                    y = np.where(x == key)[0]
                    for i in y:
                        self.img_pool[key] = list(
                            set(self.img_pool[key])
                            - set([self.cate_ind_stru["img_ind"][i]])
                        )
                if key in self.seq_subcate_exem:
                    x = np.array(self.seq_subcate_stru)
                    y = np.where(x == key)[0]
                    for i in y:
                        self.img_pool[key] = list(
                            set(self.img_pool[key])
                            - set([self.cate_ind_exem["img_ind"][i]])
                        )

            self.cate_ind_stru["img_ind_in_pool"] = []
            self.cate_ind_exem["img_ind_in_pool"] = []
            for i, cate_ind in enumerate(self.cate_ind_stru["img_ind"]):
                self.cate_ind_stru["img_ind_in_pool"].append(
                    self.image_index[self.seq_subcate_stru[i]][cate_ind]
                )
            for i, cate_ind in enumerate(self.cate_ind_exem["img_ind"]):
                self.cate_ind_exem["img_ind_in_pool"].append(
                    self.image_index[self.seq_subcate_exem[i]][cate_ind]
                )
            self.data_yml_random_log["info_exem"]["cate_ind_stru"] = self.cate_ind_stru
            self.data_yml_random_log["info_exem"]["cate_ind_exem"] = self.cate_ind_exem

        with h5py.File(
            os.path.join(
                os.path.split(__file__)[0],
                "stimuli",
                self.settings["stimuli"].get("stim_h5file"),
            ),
            "r",
        ) as h5stimfile:
            for key in h5stimfile:
                self.image_index[key].sort()
                self.images_all[key] = (
                    np.array(h5stimfile.get(key)[self.image_index[key]]) / 256
                )

    def save_yaml_random_log(self):
        if not os.path.isfile(self.yml_random_log):
            with open(self.yml_random_log, "w") as ymlseqfile:
                yaml.dump(
                    self.data_yml_random_log, ymlseqfile, default_flow_style=False
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

    def close(self):
        """Closes the experiment."""
        super().close()  # close parent class!


"""
Localizer session
Created on Sep 02, 2022
"""


class LocalizerSession(PylinkEyetrackerSession):
    def __init__(
        self, output_str, output_dir, subject, ses_nr, settings_file, eyetracker_on=True
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
        self.data_yml_random_log = {}

        # Create log folder if it does not exist
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # stimulus materials
        self.stim_file_path = os.path.join(
            os.path.split(__file__)[0],
            "stimuli",
            self.settings["stimuli"].get("stim_h5file"),
        )
        self.cue_dir_path = os.path.join(
            os.path.split(__file__)[0], "stimuli", "cue_object"
        )
        if not os.path.isfile(self.stim_file_path):
            logging.warn(
                f"Downloading stimulus file from figshare to {self.stim_file_path}"
            )
            urllib.request.urlretrieve(
                self.settings["stimuli"].get("stim_url"), self.stim_file_path
            )

        try:
            self.port = parallel.ParallelPort(address=0x0378)
            self.port.setData(0)
            self.parallel_triggering = True
        except:
            logging.warn(f"Attempted import of Parallel Port failed")
            self.parallel_triggering = False

        # set realtime mode for higher timing precision
        pylink.beginRealTimeMode(100)

        self._create_stimuli()
        self.create_trials()

    def _create_stimuli(self):
        """Creates all stimuli used in the experiment."""

        self.cue_object = self.settings["design"].get("cue_object")

        self.sequence_location = self.settings["design"].get("sequence_location")

        self.nr_images_per_category = self.settings["stimuli"].get(
            "nr_images_per_category"
        )

        # Create picture locations
        self._create_location()

        # Create masks
        self._create_mask()

        # create fixation cross
        self._create_fixation()

        with h5py.File(
            os.path.join(
                os.path.split(__file__)[0],
                "stimuli",
                self.settings["stimuli"].get("stim_h5file"),
            ),
            "r",
        ) as h5stimfile:
            self.images_all = {}
            for key in h5stimfile:
                self.images_all[key] = np.array(h5stimfile.get(key)) / 256

        # Choose the images to use for the block
        self.yml_random_log = os.path.join(
            self.output_dir,
            f"sub-{str(self.subject).zfill(2)}_ses-{str(self.ses_nr).zfill(2)}_task-pred_run-{str(self.run_nr).zfill(2)}"
            + "_random_log.yml",
        )

        if os.path.isfile(self.yml_random_log):
            with open(self.yml_random_log, "r") as ymlseqfile:
                try:
                    yml_random = yaml.safe_load(ymlseqfile)
                    self.image_index = {}
                    for key in self.images_all:
                        self.image_index[key] = yml_random.get("image_index").get(key)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            self.image_index = {}
            self.data_yml_random_log["image_index"] = {}
            for key in self.images_all:
                self.image_index[key] = rng.sample(
                    range(len(self.images_all[key])), self.nr_images_per_category
                )
                self.data_yml_random_log["image_index"][key] = self.image_index[key]

        # Create the face & scene stimuli
        self.images = {}
        self.image_stims = {}
        for key in self.images_all:
            self.images[key] = {}
            self.image_stims[key] = {}
            for loc in self.stim_location:
                self.images[key][loc] = self.images_all[key].copy()
                self.images[key][loc] = self.images[key][loc][
                    self.image_index[key]
                ]  # select the images from filepool to use
                self.image_stims[key][loc] = [
                    ImageStim(
                        win=self.win,
                        image=img,
                        mask="circle",
                        pos=self.stim_location[loc],
                        units="deg",
                        name=key + "_" + str(ind_img) + "_" + loc,
                        texRes=img.shape[1],
                        colorSpace="rgb",
                        size=self.settings["stimuli"].get("stim_size_deg"),
                        interpolate=True,
                    )
                    for ind_img, img in enumerate(self.images[key][loc])
                ]
                for ibs in self.image_stims[key][loc]:
                    ibs.draw()  # draw all the bg stimuli once, before they are used in the trials

        for loc in self.stim_location:
            self.intromask[loc].draw()

        self.beep = sound.Sound(
            os.path.join(os.path.split(__file__)[0], "stimuli", "beep.wav")
        )
        self.win.flip()
        self.win.flip()

    def _create_location(self):
        """Creates the location of the sequence"""
        self.distance_from_center = self.settings["stimuli"].get(
            "distance_from_center"
        ) * math.sqrt(1 / 2)
        self.stim_location = self.settings["design"].get("stim_location")
        for key in self.stim_location:
            self.stim_location[key] = [
                x * self.distance_from_center for x in self.stim_location[key]
            ]

    def _create_fixation(self):
        self.fixation_cue = {}
        self.fixation_w = FixationBullsEye(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("fixation_size_pixels"),
            color=(0.5, 0.5, 0.5, 1),
            **{"lineWidth": self.settings["stimuli"].get("outer_fix_linewidth")},
        )
        for cueobject in self.cue_object:
            img = os.path.join(self.cue_dir_path, cueobject + ".png")
            self.fixation_cue[cueobject] = ImageStim(
                win=self.win,
                image=img,
                pos=[0, 0],
                units="deg",
                name=cueobject,
                texRes=256,
                colorSpace="rgb",
                size=self.settings["stimuli"].get("cue_size_deg"),
                interpolate=True,
            )
            self.fixation_cue[cueobject].draw()

        self.fixation_dot = FixationCue(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("cue_size_pixels"),
            color=-1,
            **{"lineWidth": self.settings["stimuli"].get("outer_fix_linewidth")},
        )
        self.filled_circles = PlaceHolderCircles(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("stim_size_deg"),
            color=-0.5,
            ecc=self.settings["stimuli"].get("distance_from_center"),
            linewidth=self.settings["stimuli"].get("stim_circle_linewidth"),
        )
        self.filled_circles.draw()
        self.empty_circles = PlaceHolderCircles(
            win=self.win,
            circle_radius=self.settings["stimuli"].get("stim_size_deg"),
            color=-0.5,
            ecc=self.settings["stimuli"].get("distance_from_center"),
            linewidth=self.settings["stimuli"].get("stim_circle_linewidth"),
            fill=False,
        )
        self.empty_circles.draw()
        for loc in self.stim_location:
            self.intromask[loc].draw()
        self.fixation_dot.draw()
        self.win.flip()
        self.win.flip()

    def _create_mask(self):
        self.intromask = {}
        for loc in self.stim_location:
            self.intromask[loc] = GratingStim(
                self.win,
                tex=np.ones((4, 4)),
                pos=self.stim_location[loc],
                units="deg",
                contrast=1,
                color=(0.0, 0.0, 0.0),
                name="intromask" + "_" + loc,
                size=self.settings["stimuli"].get("stim_size_deg"),
            )

    def create_trials(self):
        self.instruction_text = eval(
            f"f'{self.settings['stimuli'].get('instruction_text')}'"
        )

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

        # Create phase_durations
        self.phase_durations = []
        phase_durations = [self.settings["stimuli"].get("stim_refresh_time")]

        # Create the trial sequence
        self.nr_comb = self.settings["design"].get("nr_per_combination_localizer")
        self.nr_loc = len(self.settings["design"].get("location_name"))
        self.nr_cate = len(self.settings["design"].get("category_key"))
        self.nr_trials = self.nr_loc * self.nr_cate * self.nr_comb
        self.seq_cate = np.repeat(
            self.settings["design"].get("category_key"), self.nr_loc * self.nr_comb
        )
        self.seq_subcate = np.repeat(
            self.settings["design"].get("category").get("face"),
            self.nr_loc * self.nr_comb / 2,
        )
        self.seq_subcate = np.concatenate(
            [
                self.seq_subcate,
                np.repeat(
                    self.settings["design"].get("category").get("scene"),
                    self.nr_loc * self.nr_comb / 2,
                ),
            ]
        )
        self.seq_loc = np.tile(
            self.settings["design"].get("location_name"), self.nr_cate * self.nr_comb
        )
        self.seq_cate_ind = []
        [
            self.seq_cate_ind.append(
                rng.sample(range(self.nr_images_per_category), 1)[0]
            )
            for _ in range(self.nr_trials)
        ]

        """Insert the objects trials"""
        self.nr_per_object = self.nr_cate * self.nr_comb
        self.list_object = np.repeat(self.cue_object, self.nr_per_object)

        self.nr_trials = self.nr_trials + len(self.list_object)
        for obj in self.list_object:
            self.seq_cate = np.append(self.seq_cate, "object")
            self.seq_subcate = np.append(self.seq_subcate, obj)
            self.seq_loc = np.append(self.seq_loc, "center")
            self.seq_cate_ind = np.append(self.seq_cate_ind, -1)

        """permutation"""
        p1 = np.random.permutation(self.nr_trials)
        self.seq_cate = self.seq_cate[p1]
        self.seq_subcate = self.seq_subcate[p1]
        self.seq_loc = self.seq_loc[p1]
        self.seq_cate_ind = self.seq_cate_ind[p1]

        for _ in range(self.nr_trials):
            x = phase_durations.copy()
            x.append(rng.sample(self.settings["stimuli"].get("ITI_time"), 1)[0])
            self.phase_durations.append(x)

        """Create trials"""
        for _ in [*range(self.nr_trials)]:
            parameters = {'trial_ind':   int(self.trial_counter-2),
                          'location':    self.seq_loc[self.trial_counter-2],
                          'category':    self.seq_cate[self.trial_counter-2],
                          'subcategory': self.seq_subcate[self.trial_counter-2],
                          'img_ind':     self.seq_cate_ind[self.trial_counter-2],
                          'type_test':   self.type_test[self.trial_counter-2]
                          }

            self.trials.append(
                LocalizerTrial(
                    session=self,
                    trial_nr=self.trial_counter,
                    phase_durations=self.phase_durations[self.trial_counter - 2],
                    phase_names=["stim", "ITI"],
                    parameters=parameters,
                    timing="seconds",
                    verbose=True,
                    draw_each_frame=False,
                )
            )
            self.trial_counter += 1

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

    def close(self):
        """Closes the experiment."""
        super().close()  # close parent class!
