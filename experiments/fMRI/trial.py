#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

from exptools2.core import Trial
from psychopy.core import getTime
from psychopy.visual import TextStim, ImageStim


class TestTrial(Trial):
    """
    Initializes a SequenceTrial object.

    Parameters
    ----------
    session : exptools Session object
        A Session object (needed for metadata)
    trial_nr: int
        Trial nr of trial
    phase_durations : array-like
        List/tuple/array with phase durations
    phase_names : array-like
        List/tuple/array with names for phases (only for logging),
        optional (if None, all are named 'stim')
    parameters : dict
        Dict of parameters that needs to be added to the log of this trial
    keys : array-like
        List/tuple/array with keys that can be pressed
    timing : str
        The "units" of the phase durations. Default is 'seconds', where we
        assume the phase-durations are in seconds. The other option is
        'frames', where the phase-"duration" refers to the number of frames.
    verbose : bool
        Whether to print extra output (mostly timing info)
    """

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        parameters,
        keys,
        timing,
        verbose=True,
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            parameters,
            timing,
            load_next_during_phase=None,
            verbose=verbose,
            draw_each_frame=draw_each_frame,
        )
        self.keys = keys
        self.freq = round(
            (1 / 15) * 1 / self.session.win.monitorFramePeriod
        )  # 15 Hz of flickering for fixation dot

    def draw(self):
        if self.phase == 0:
            self.session.fixbullseye.draw()
            self.session.fixation_dot_flk.draw()
        elif self.phase == 1:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.gabors[(45, 0)].draw()
            self.session.gabors[(135, 45)].draw()
            self.session.gabors[(225, 0)].draw()
            self.session.gabors[(315, 45)].draw()
            self.session.win.flip()
        elif self.phase == 2:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.win.flip()

    def get_events(self):
        events = super().get_events()
        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()

    def run(self):
        super().run()


class TaskTrial(Trial):
    """
    Initializes a SequenceTrial object.

    Parameters
    ----------
    session : exptools Session object
        A Session object (needed for metadata)
    trial_nr: int
        Trial nr of trial
    phase_durations : array-like
        List/tuple/array with phase durations
    phase_names : array-like
        List/tuple/array with names for phases (only for logging),
        optional (if None, all are named 'stim')
    parameters : dict
        Dict of parameters that needs to be added to the log of this trial
    keys : array-like
        List/tuple/array with keys that can be pressed
    timing : str
        The "units" of the phase durations. Default is 'seconds', where we
        assume the phase-durations are in seconds. The other option is
        'frames', where the phase-"duration" refers to the number of frames.
    verbose : bool
        Whether to print extra output (mostly timing info)
    """

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        parameters,
        keys,
        timing,
        show_background_gabors=True,
        background_gabor_angles=None,
        verbose=True,
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            parameters,
            timing,
            load_next_during_phase=None,
            verbose=verbose,
            draw_each_frame=draw_each_frame,
        )
        self.keys = keys
        self.freq = round(
            (1 / self.session.settings["stimuli"].get("fixation_temporal_freq"))
            * 1
            / self.session.win.monitorFramePeriod
        )  # set flickering rate for fixation dot
        self.show_background_gabors = show_background_gabors
        self.background_gabor_angles = background_gabor_angles

    def draw(self):
        if self.phase == 0:
            self.session.fixbullseye.draw()
            self.session.fixation_dot_flk.draw()
            self.session.win.flip()
        elif self.phase == 1:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.gabors[
                (
                    self.parameters["angle_T"],
                    self.parameters["ori_T"],
                    self.parameters["color_T"],
                )
            ].draw()
            if self.parameters["angle_D"] is not None:
                self.session.gabors[
                    (
                        self.parameters["angle_D"],
                        self.parameters["ori_D"],
                        self.parameters["color_D"],
                    )
                ].draw()

            for bg_angle in self.background_gabor_angles:
                self.session.gabors[
                    (
                        bg_angle,
                        int((self.parameters["ori_D"] + 90) % 180),
                        self.parameters["color_T"],
                    )
                ].draw()

            self.session.win.flip()
            self.ITI_start_time = self.session.clock.getTime()
        elif self.phase == 2:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.win.flip()

    def get_events(self):
        events = super().get_events()
        if self.phase == 2:
            if events is not None:
                for key, t in events:
                    if (
                        self.trial_nr + 1 - self.session.nr_instruction_trials
                    ) % 4 == 0:
                        if (key == self.session.mri_trigger) & (
                            t - self.ITI_start_time
                            > (
                                self.session.settings["design"].get("task_ITI_time")
                                - 0.1
                            )
                        ):
                            self.stop_phase()

    def run(self):
        super().run()


class TaskTrial_train(TaskTrial):
    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        parameters,
        keys,
        timing,
        show_background_gabors=True,
        background_gabor_angles=None,
        verbose=True,
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            parameters,
            keys,
            timing,
            show_background_gabors=True,
            verbose=verbose,
            draw_each_frame=draw_each_frame,
        )
        self.key = None
        self.resp_ahead = False
        self.show_background_gabors = show_background_gabors
        self.background_gabor_angles = background_gabor_angles

    def draw(self):
        if self.phase == 0:
            self.session.fixbullseye.draw()
            self.session.fixation_dot_flk.draw()
            self.session.win.flip()
        elif self.phase == 1:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            if self.show_background_gabors:
                for bg_angle in self.background_gabor_angles:
                    self.session.gabors[
                        (
                            bg_angle,
                            int((self.parameters["ori_D"] + 90) % 180),
                            self.parameters["color_T"],
                        )
                    ].draw()
            self.session.gabors[
                (
                    self.parameters["angle_T"],
                    self.parameters["ori_T"],
                    self.parameters["color_T"],
                )
            ].draw()
            if self.parameters["angle_D"] is not None:
                self.session.gabors[
                    (
                        self.parameters["angle_D"],
                        self.parameters["ori_D"],
                        self.parameters["color_D"],
                    )
                ].draw()
            if self.key == self.parameters["corr_key"]:
                self.session.fixation_dot.inner_circle.color = "chartreuse"
                self.session.resp_task[self.parameters["ind_TaskTrial"]] = True
            elif self.key is None:
                pass
            else:
                self.session.fixation_dot.inner_circle.color = "red"
            self.session.win.flip()
        elif self.phase == 2:
            if self.resp_ahead:
                self.stop_phase()
            if self.key == self.parameters["corr_key"]:
                self.session.fixation_dot.inner_circle.color = "chartreuse"
                self.session.resp_task[self.parameters["ind_TaskTrial"]] = True
            elif self.key is None:
                pass
            else:
                self.session.fixation_dot.inner_circle.color = "red"
            self.session.fixation_dot.inner_circle.contrast = self.session.settings[
                "stimuli"
            ].get("stim_gabor_contrast")
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.win.flip()
        elif self.phase == 3:
            if self.key == self.parameters["corr_key"]:
                self.session.fixation_dot.inner_circle.color = "chartreuse"
                self.session.resp_task[self.parameters["ind_TaskTrial"]] = True
            else:
                self.session.fixation_dot.inner_circle.color = "red"
                self.session.resp_task[self.parameters["ind_TaskTrial"]] = False
            self.session.fixation_dot.inner_circle.contrast = self.session.settings[
                "stimuli"
            ].get("stim_gabor_contrast")
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.win.flip()
            self.session.fixation_dot.inner_circle.fillColor = -1
            self.session.fixation_dot.inner_circle.lineColor = -1
            self.session.fixation_dot.inner_circle.contrast = self.session.settings[
                "stimuli"
            ].get("stim_gabor_contrast")

    def get_events(self):
        events = super(TaskTrial, self).get_events()
        if self.phase == 2 or self.phase == 1:
            if self.keys is None:
                if events:
                    self.stop_phase()
            else:
                for key, t in events:
                    if key in self.keys:
                        self.key = key
                        self.resp_ahead = True
                        self.stop_phase()


class PingTrial(Trial):
    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        parameters,
        keys,
        timing,
        verbose=True,
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            parameters,
            timing,
            load_next_during_phase=None,
            verbose=verbose,
            draw_each_frame=draw_each_frame,
        )
        self.keys = keys
        self.freq = round(
            (1 / self.session.settings["stimuli"].get("fixation_temporal_freq"))
            * 1
            / self.session.win.monitorFramePeriod
        )  # set flickering rate for fixation dot

    def draw(self):
        if self.phase == 0:
            self.session.fixbullseye.draw()
            self.session.fixation_dot_flk.draw()
            self.session.win.flip()
        elif self.phase == 1:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.checkerboards[
                (
                    self.parameters["angle_Ping"],
                    self.parameters["ori_Ping"],
                    self.parameters["color_Ping"],
                )
            ].draw()
            self.session.win.flip()
            self.ITI_start_time = self.session.clock.getTime()
        elif self.phase == 2:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.win.flip()

    def get_events(self):
        events = super().get_events()
        if self.phase == 2 or self.phase == 1:
            if self.session.ses_nr in ["practice", "train"]:
                if self.keys is None:
                    if events:
                        pass
                else:
                    for key, t in events:
                        if key in self.keys:
                            self.session.resp_ping = np.append(
                                self.session.resp_ping, 0
                            )
            elif self.session.ses_nr == "test":
                if self.phase == 2:
                    if events is not None:
                        for key, t in events:
                            if (
                                self.trial_nr + 1 - self.session.nr_instruction_trials
                            ) % 4 == 0:
                                if (key == self.session.mri_trigger) & (
                                    t - self.ITI_start_time
                                    > (
                                        self.session.settings["design"].get(
                                            "ping_ITI_time"
                                        )
                                        - 0.1
                                    )
                                ):
                                    self.stop_phase()

    def run(self):
        super().run()


class RestingTrial(Trial):
    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        parameters,
        keys,
        timing,
        verbose=True,
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            parameters,
            timing,
            load_next_during_phase=None,
            verbose=verbose,
            draw_each_frame=draw_each_frame,
        )
        self.keys = keys

    def draw(self):
        if self.phase == 0:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.win.flip()
        elif self.phase == 1:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.win.flip()
            self.ITI_start_time = self.session.clock.getTime()
        elif self.phase == 2:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.win.flip()

    def get_events(self):
        events = super().get_events()
        if self.phase == 2:
            if events is not None:
                for key, t in events:
                    if (
                        self.trial_nr + 1 - self.session.nr_instruction_trials
                    ) % 4 == 0:
                        if (key == self.session.mri_trigger) & (
                            t - self.ITI_start_time
                            > (
                                self.session.settings["design"].get("ping_ITI_time")
                                - 0.1
                            )
                        ):
                            self.stop_phase()

    def run(self):
        super().run()


class SuckerTrial(Trial):
    """ """

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        parameters,
        keys,
        timing,
        verbose=True,
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            parameters,
            timing,
            load_next_during_phase=None,
            verbose=verbose,
            draw_each_frame=draw_each_frame,
        )
        self.keys = keys
        self.freq = round(
            (1 / self.session.settings["stimuli"].get("fixation_temporal_freq"))
            * 1
            / self.session.win.monitorFramePeriod
        )  # set flickering rate for fixation dot

    def draw(self):
        if self.phase == 0:
            self.session.fixbullseye.draw()
            self.session.fixation_dot_flk.draw()
        elif self.phase == 1:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.draw()
            self.session.win.flip()
            self.ITI_start_time = self.session.clock.getTime()
        elif self.phase == 2:
            self.session.fixbullseye.draw()
            self.session.fixation_dot.inner_circle.opacity = 1
            self.session.fixation_dot.outer_circle.opacity = 1
            self.session.fixation_dot.draw()
            self.session.win.flip()

    def get_events(self):
        events = super().get_events()
        if self.phase == 2:
            if events is not None:
                for key, t in events:
                    if (
                        self.trial_nr + 1 - self.session.nr_instruction_trials
                    ) % 4 == 0:
                        if (key == self.session.mri_trigger) & (
                            t - self.ITI_start_time
                            > (
                                self.session.settings["design"].get("ping_ITI_time")
                                - 0.1
                            )
                        ):
                            self.stop_phase()

    def run(self):
        super().run()


class InstructionTrial(Trial):
    """Simple trial with instruction text."""

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations=[np.inf],
        phase_names=["Instruction"],
        txt=None,
        keys=None,
        txt_height=0.3,
        txt_width=15,
        txt_position_x=0,
        txt_position_y=0.5,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )

        if txt is None:
            txt = """Press any button to continue."""

        self.text = TextStim(
            self.session.win,
            txt,
            height=txt_height,
            wrapWidth=txt_width,
            units="deg",
            pos=[txt_position_x, txt_position_y],
            font="Arial",
            alignText="center",
            anchorHoriz="center",
            anchorVert="center",
        )
        self.text.setSize(txt_height)
        self.keys = keys

    def draw(self):
        self.session.fixbullseye.draw()
        self.session.fixation_dot.draw()
        self.text.draw()
        self.session.win.flip()

    def get_events(self):
        events = super().get_events()
        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()


class DummyWaiterTrial(InstructionTrial):
    """Simple trial with text (trial x) and fixation."""

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations=None,
        phase_names=None,
        txt="Waiting for scanner triggers.",
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            txt,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )

    def draw(self):
        self.session.fixbullseye.draw()
        self.session.fixation_dot.draw()
        if self.phase == 0:
            self.text.draw()
        else:
            pass
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)
        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    if self.phase == 0:
                        self.stop_phase()
                        #####################################################
                        ## TRIGGER HERE
                        #####################################################
                        self.session.experiment_start_time = getTime()


class WaitStartTriggerTrial(Trial):
    def __init__(
        self,
        session,
        trial_nr,
        phase_durations=[np.inf],
        phase_names=["waiting_start_trigger"],
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            draw_each_frame=draw_each_frame,
        )

    def draw(self):
        self.session.fixbullseye.draw()
        self.session.fixation_dot.draw()
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)
        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    self.stop_phase()
                    self.session.experiment_start_time = getTime()


class OutroTrial(Trial):
    """Simple trial with only fixation cross."""

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )

    def draw(self):
        self.session.fixbullseye.draw()
        self.session.fixation_dot.draw()
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)
        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    pass


class FeedbackTrial(Trial):
    """Simple trial with feedback text."""

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations=[np.inf],
        txt=None,
        keys=None,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )
        self.keys = keys

    def draw(self):
        self.session.fixbullseye.draw()
        self.session.fixation_dot.draw()
        txt_height = self.session.settings["various"].get("text_height")
        txt_width = self.session.settings["various"].get("text_width")
        text_position_x = self.session.settings["various"].get("text_position_x")
        text_position_y = (
            self.session.settings["various"].get("text_position_y")
            + self.session.roll_dist
        )

        ACC = np.mean(self.session.resp_task) * 100

        if self.session.ses_nr in ["practice", "train"]:
            txt_0 = f"This is Session-{self.session.ses_nr} Run-{self.session.run_nr}, Your score in this block is {str(math.ceil(ACC))}%."
        elif self.session.ses_nr == "test":
            txt_0 = f"This is Session-{self.session.ses_nr} Run-{self.session.run_nr}."

        txt_1 = f"Please ask the experimenter to continue..."
        self.text_0 = TextStim(
            self.session.win,
            txt_0,
            height=txt_height,
            wrapWidth=txt_width,
            units="deg",
            pos=[text_position_x, text_position_y],
            font="Arial",
            alignText="center",
            anchorHoriz="center",
            anchorVert="center",
        )
        self.text_1 = TextStim(
            self.session.win,
            txt_1,
            height=txt_height,
            wrapWidth=txt_width,
            pos=[text_position_x, text_position_y - 1],
            units="deg",
            font="Arial",
            alignText="center",
            anchorHoriz="center",
            anchorVert="center",
        )
        self.text_0.setSize(txt_height)
        self.text_1.setSize(txt_height)
        self.text_0.draw()
        self.text_1.draw()
        self.session.win.flip()

    def get_events(self):
        events = super().get_events()

        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()


class RollDownTheWindowTrial(Trial):
    def __init__(
        self,
        session,
        trial_nr,
        phase_durations=[np.inf],
        keys=None,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )
        self.keys = keys

    def draw(self):
        self.session.fixbullseye.draw()
        self.session.fixation_dot.draw()
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)
        if events:
            for key, t in events:
                if key == self.keys[0]:
                    self.session.roll_dist += self.session.settings["design"].get(
                        "roll_dist_step"
                    )
                elif key == self.keys[1]:
                    self.session.roll_dist -= self.session.settings["design"].get(
                        "roll_dist_step"
                    )
                self.session.fixbullseye.circle1.pos = (0, self.session.roll_dist)
                self.session.fixbullseye.circle2.pos = (0, self.session.roll_dist)
                self.session.fixation_dot.outer_circle.pos = (0, self.session.roll_dist)
                self.session.fixation_dot.inner_circle.pos = (0, self.session.roll_dist)
                self.session.fixation_dot.line1.start = (
                    -self.session.fixation_dot.circle_radius
                    + self.session.fixation_dot.pos[0],
                    self.session.roll_dist,
                )
                self.session.fixation_dot.line1.end = (
                    self.session.fixation_dot.circle_radius
                    + self.session.fixation_dot.pos[0],
                    self.session.roll_dist,
                )
                self.session.fixation_dot.line2.start = (
                    0,
                    -self.session.fixation_dot.circle_radius + self.session.roll_dist,
                )
                self.session.fixation_dot.line2.end = (
                    0,
                    self.session.fixation_dot.circle_radius + self.session.roll_dist,
                )
                self.session.data_yml_log["window"] = {
                    "roll_dist": self.session.roll_dist,
                }
                self.session.win.flip()
                self.session.save_yaml_log()


class PingpRFTrial(Trial):
    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        parameters,
        keys,
        timing,
        verbose=True,
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            parameters,
            timing,
            verbose=verbose,
            draw_each_frame=draw_each_frame,
        )
        self.keys = keys

    def draw(self):
        self.session.fixbullseye.draw()
        if self.phase == 0:
            self.session.fixation_dot_flk.draw()
            self.session.win.flip()
        elif self.phase == 1:
            self.session.fixation_dot.draw()
            self.session.checkerboards[
                (
                    self.parameters["angle_Ping"],
                    self.parameters["ori_Ping"],
                    self.parameters["direction"],
                    self.parameters["color_Ping"],
                )
            ].draw()
            self.session.win.flip()
            self.ITI_start_time = self.session.clock.getTime()
        elif self.phase == 2:
            self.session.fixation_dot.draw()
            self.session.win.flip()

    def get_events(self):
        events = super().get_events()
        if self.phase == 2:
            if events is not None:
                for key, t in events:
                    if (
                        self.trial_nr + 1 - self.session.nr_instruction_trials
                    ) % 4 == 0:
                        if (key == self.session.mri_trigger) & (
                            t - self.ITI_start_time
                            > (
                                self.session.settings["design"].get("ping_ITI_time")
                                - 0.1
                            )
                        ):
                            self.stop_phase()

    def run(self):
        super().run()


class PingpRFTrial_train(Trial):
    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        parameters,
        keys,
        timing,
        verbose=True,
        draw_each_frame=False,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            parameters,
            timing,
            verbose=verbose,
            draw_each_frame=draw_each_frame,
        )
        self.keys = keys
        self.key = None

    def draw(self):
        self.session.fixbullseye.draw()
        if self.phase == 0:
            self.session.fixation_dot_flk.draw()
            self.session.win.flip()
        elif self.phase == 1:
            self.session.fixation_dot.draw()
            self.session.checkerboards[
                (
                    self.parameters["angle_Ping"],
                    self.parameters["ori_Ping"],
                    self.parameters["direction"],
                    self.parameters["color_Ping"],
                )
            ].draw()
            self.session.win.flip()
        elif self.phase == 2:
            self.session.fixation_dot.draw()
            self.session.win.flip()
        elif self.phase == 3:
            if self.key == self.parameters["corr_key"]:
                self.session.fixation_dot.inner_circle.color = "chartreuse"
                self.session.resp_task[self.parameters["ind_TaskTrial"]] = True
            else:
                self.session.fixation_dot.inner_circle.color = "red"
                self.session.resp_task[self.parameters["ind_TaskTrial"]] = False
            self.session.fixation_dot.inner_circle.contrast = self.session.settings[
                "stimuli"
            ].get("stim_gabor_contrast")
            self.session.fixation_dot.draw()
            self.session.win.flip()
            self.session.fixation_dot.inner_circle.fillColor = -1
            self.session.fixation_dot.inner_circle.lineColor = -1
            self.session.fixation_dot.inner_circle.contrast = self.session.settings[
                "stimuli"
            ].get("stim_gabor_contrast")

    def get_events(self):
        events = super().get_events()
        if self.phase == 2 or self.phase == 1:
            if self.keys is None:
                if events:
                    self.stop_phase()
            else:
                for key, t in events:
                    if key in self.keys:
                        self.key = key
                        self.stop_phase()


class AwarenessCheckTrial(Trial):
    """
    Participants are asked to answer whether the highlighted location contains the HPL of distractor or not.
    """

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        parameters,
        keys=None,
        txt=None,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            parameters,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )
        self.keys = keys
        self.highlighted = parameters["highlighted"]

        # text
        if txt is None and len(self.highlighted) == 2:
            self.txt = self.session.settings["stimuli"].get("awareness_text_paired")
        elif txt is None and len(self.highlighted) == 1:
            self.txt = self.session.settings["stimuli"].get("awareness_text_single")
        else:
            self.txt = txt

        txt_height = (
            self.session.settings["various"].get("awareness_check").get("text_height")
        )
        txt_width = (
            self.session.settings["various"].get("awareness_check").get("text_width")
        )
        text_position_x = (
            self.session.settings["various"]
            .get("awareness_check")
            .get("text_awarenesscheck_position_x")
        )
        text_position_y = (
            self.session.settings["various"]
            .get("awareness_check")
            .get("text_awarenesscheck_position_y")
            + self.session.roll_dist
        )

        self.text = TextStim(
            self.session.win,
            self.txt,
            height=txt_height,
            wrapWidth=txt_width,
            units="deg",
            pos=[text_position_x, text_position_y],
            font="Arial",
            alignText="center",
            anchorHoriz="center",
            anchorVert="center",
        )
        self.text.setSize(txt_height)

    def draw(self):
        self.session.fixbullseye.draw()
        self.session.fixation_dot.draw()
        for _, placeholder in self.session.placeholders.items():
            placeholder.draw()

        for angle in self.highlighted:
            self.session.highlighters_qm[angle].draw()

        self.text.draw()
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)
        if events:
            for key, t in events:
                if key in self.keys:
                    if key == self.keys[0]:
                        pass
                    elif key == self.keys[1]:
                        pass
                    self.stop_trial()
                else:
                    pass


class InstructionTrial_awareness(InstructionTrial):
    """Simple trial with instruction text."""

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations=[np.inf],
        txt=None,
        keys=None,
        txt_height=0.3,
        txt_width=15,
        txt_position_x=0,
        txt_position_y=0.5,
        image=None,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations=phase_durations,
            txt=txt,
            keys=keys,
            txt_height=txt_height,
            txt_width=txt_width,
            txt_position_x=txt_position_x,
            txt_position_y=txt_position_y,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )
        self.instruction_picture = ImageStim(
            self.session.win,
            image,
            units="deg",
            pos=[
                self.session.settings["various"].get("instruction_image_position_x"),
                self.session.settings["various"].get("instruction_image_position_y"),
            ],
            size=self.session.settings["various"].get("instruction_image_size"),
        )
        self.keys = keys

    def draw(self):
        self.session.fixbullseye.draw()
        self.session.fixation_dot.draw()
        self.text.draw()
        self.instruction_picture.draw()
        self.session.win.flip()


class AwarenessRateTrial(Trial):
    """
    Participants are asked to Rate the possibility of the highlighted location contains the HPL of distractor or not (0-5).
    """

    def __init__(
        self,
        session,
        trial_nr,
        phase_durations,
        phase_names,
        parameters,
        keys=None,
        txt=None,
        draw_each_frame=False,
        **kwargs,
    ):
        super().__init__(
            session,
            trial_nr,
            phase_durations,
            phase_names,
            parameters,
            draw_each_frame=draw_each_frame,
            **kwargs,
        )
        self.keys = keys
        self.rating_angle = parameters["angle"]

        # text
        self.txt = self.session.settings["stimuli"].get("awareness_text_rate")
        self.txt_operation = self.session.settings["stimuli"].get(
            "awareness_text_rate_operation"
        )
        self.txt_height = (
            self.session.settings["various"].get("awareness_rate").get("text_height")
        )
        self.txt_width = (
            self.session.settings["various"].get("awareness_rate").get("text_width")
        )
        text_position_x = (
            self.session.settings["various"]
            .get("awareness_rate")
            .get("text_awarenesscheck_position_x")
        )
        text_position_y = (
            self.session.settings["various"]
            .get("awareness_rate")
            .get("text_awarenesscheck_position_y")
            + self.session.roll_dist
        )

        self.text = TextStim(
            self.session.win,
            self.txt,
            height=self.txt_height,
            wrapWidth=self.txt_width,
            units="deg",
            pos=[text_position_x, text_position_y],
            font="Arial",
            alignText="center",
            anchorHoriz="center",
            anchorVert="center",
        )
        self.text.setSize(self.txt_height)
        self.stop_confirmed = False

    def draw(self):
        self.text_operation = TextStim(
            self.session.win,
            self.txt_operation,
            height=self.txt_height,
            wrapWidth=self.txt_width,
            units="deg",
            pos=[
                self.session.settings["various"]
                .get("awareness_rate")
                .get("text_operation_position_x"),
                self.session.settings["various"]
                .get("awareness_rate")
                .get("text_operation_position_y")
                + self.session.roll_dist,
            ],
            font="Arial",
            alignText="center",
            anchorHoriz="center",
            anchorVert="center",
        )

        self.session.fixbullseye.draw()
        self.session.fixation_dot.draw()
        for _, placeholder in self.session.placeholders.items():
            placeholder.draw()

        self.session.highlighters[self.rating_angle].draw()

        for angle in self.session.awareness_rating.keys():
            self.session.rate_numbers[
                angle, self.session.awareness_rating[angle]
            ].draw()

        self.text.draw()
        self.text_operation.draw()
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)
        if events:
            for key, t in events:
                if key in self.keys:
                    if key == self.keys[0]:
                        if self.stop_confirmed == True:
                            self.stop_confirmed = False
                            self.txt_operation = self.session.settings["stimuli"].get(
                                "awareness_text_rate_operation"
                            )
                        elif (self.stop_confirmed == False) & (
                            self.session.awareness_rating[self.rating_angle]
                            < self.session.settings["design"].get(
                                "awareness_rate_range"
                            )[1]
                        ):
                            self.session.awareness_rating[self.rating_angle] += 1
                            self.txt_operation = self.session.settings["stimuli"].get(
                                "awareness_text_rate_operation"
                            )
                        elif (self.stop_confirmed == False) & (
                            self.session.awareness_rating[self.rating_angle]
                            == self.session.settings["design"].get(
                                "awareness_rate_range"
                            )[1]
                        ):
                            self.session.awareness_rating[self.rating_angle] = (
                                self.session.settings["design"].get(
                                    "awareness_rate_range"
                                )[0]
                            )
                            self.txt_operation = self.session.settings["stimuli"].get(
                                "awareness_text_rate_reset"
                            )
                    elif key == self.keys[1]:
                        if (self.stop_confirmed == True) & (
                            self.session.awareness_rating[self.rating_angle] > 0
                        ):
                            self.stop_trial()
                        elif self.session.awareness_rating[self.rating_angle] == 0:
                            self.txt_operation = self.session.settings["stimuli"].get(
                                "awareness_text_rate_need_operation"
                            )
                            self.stop_confirmed = False
                        else:
                            self.stop_confirmed = True
                            self.txt_operation = self.session.settings["stimuli"].get(
                                "awareness_text_rate_confirm"
                            )

                else:
                    pass
        self.parameters[self.rating_angle] = self.session.awareness_rating[
            self.rating_angle
        ]
