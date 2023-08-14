#!/usr/bin/env python
#-*- coding: utf-8 -*-

import math, time
import numpy as np
import pandas as pd

from exptools2.core import Trial
from psychopy.core import getTime
from psychopy.visual import TextStim, ImageStim
from psychopy import logging
from stimuli import FixationBullsEye


class SequenceTrial(Trial):
    
    def __init__(self, session, trial_nr, phase_durations, phase_names,
                 parameters, timing, verbose=True, draw_each_frame=False):
        """ Initializes a BarPassTrial object. 
        
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
        timing : str
            The "units" of the phase durations. Default is 'seconds', where we
            assume the phase-durations are in seconds. The other option is
            'frames', where the phase-"duration" refers to the number of frames.
        verbose : bool
            Whether to print extra output (mostly timing info)
        """
        super().__init__(session, trial_nr, phase_durations, phase_names,
                         parameters, timing, load_next_during_phase=None, verbose=verbose, draw_each_frame=draw_each_frame)
        self.count = 0
        self.idx = None
        if 'space' in self.parameters['correct_kb']:
            self.ind_res = self.parameters['correct_kb'].index('space')
            self.session.res_feedback[self.trial_nr] = -1
        else:
            self.ind_res = None

    def draw(self):
        ind_fixation = self.phase%2
        self.session.fixation_w.draw()
        self.session.fixation_dot.draw()
        self.session.filled_circles.draw()
        color = 'colored'
        if self.phase == 0:
            self.session.fixation_cue[self.parameters['cue_object']].draw()
        elif ind_fixation == 0 and self.phase != len(self.phase_names)-1 and self.phase != len(self.phase_names)-2:
            color_prev, color_next = 'colored', 'colored'
            if self.parameters['AT_task'][int((self.phase-2)/2)] == True:
                color_prev = 'greyscaled'
            elif self.parameters['AT_task'][int((self.phase)/2)] == True:
                color_next = 'greyscaled'
            if self.session.ses_nr == 0 or self.session.ses_nr == 2:
                self.session.image_stims[color_prev][self.session.seq_subcate[self.parameters['trial_ind']][int(
                    (self.phase-2)/2)]][self.session.seq_location[self.parameters['trial_ind']][int(
                        (self.phase-2)/2)]][self.session.seq_cate_ind[self.parameters['trial_ind']][int((self.phase-2)/2)]].draw()
                self.session.image_stims[color_next][self.session.seq_subcate[self.parameters['trial_ind']][int(
                    (self.phase)/2)]][self.session.seq_location[self.parameters['trial_ind']][int(
                        (self.phase)/2)]][self.session.seq_cate_ind[self.parameters['trial_ind']][int((self.phase)/2)]].draw()
            elif self.session.ses_nr == 1:
                if int((self.phase-2)/2)>=self.session.ind_test[self.parameters['trial_ind']]:
                    # self.session.intromask[self.session.seq_location[self.parameters['trial_ind']][int(
                    #     (self.phase-2)/2)]].draw()
                    # self.session.intromask[self.session.seq_location[self.parameters['trial_ind']][int(
                    #     (self.phase)/2)]].draw()
                    pass
                else:
                    self.session.image_stims[color_prev][self.session.seq_subcate[self.parameters['trial_ind']][int(
                        (self.phase-2)/2)]][self.session.seq_location[self.parameters['trial_ind']][int(
                            (self.phase-2)/2)]][self.session.seq_cate_ind[self.parameters['trial_ind']][int((self.phase-2)/2)]].draw()
                    self.session.image_stims[color_next][self.session.seq_subcate[self.parameters['trial_ind']][int(
                        (self.phase)/2)]][self.session.seq_location[self.parameters['trial_ind']][int(
                            (self.phase)/2)]][self.session.seq_cate_ind[self.parameters['trial_ind']][int((self.phase)/2)]].draw()
        elif ind_fixation != 0 and self.phase != len(self.phase_names)-1:
            if self.parameters['AT_task'][int((self.phase-1)/2)] == True:
                    color = 'greyscaled'
            if self.session.ses_nr == 0 or self.session.ses_nr == 2:
                self.session.image_stims[color][self.session.seq_subcate[self.parameters['trial_ind']][int(
                    (self.phase-1)/2)]][self.session.seq_location[self.parameters['trial_ind']][int(
                        (self.phase-1)/2)]][self.session.seq_cate_ind[self.parameters['trial_ind']][int((self.phase-1)/2)]].draw()
            elif self.session.ses_nr == 1:
                if int((self.phase-1)/2)>=self.session.ind_test[self.parameters['trial_ind']]+1:
                    # self.session.intromask[self.session.seq_location[self.parameters['trial_ind']][int(
                    #     (self.phase-1)/2)]].draw()
                    # self.session.intromask.draw()
                    pass
                else:
                    self.session.image_stims[color][self.session.seq_subcate[self.parameters['trial_ind']][int(
                        (self.phase-1)/2)]][self.session.seq_location[self.parameters['trial_ind']][int(
                            (self.phase-1)/2)]][self.session.seq_cate_ind[self.parameters['trial_ind']][int((self.phase-1)/2)]].draw()
        elif self.phase == len(self.phase_names)-1 or self.phase == len(self.phase_names)-2:
            pass
        else:
            raise ValueError('Exceeded phase number')
        self.session.empty_circles.draw()
        self.session.win.flip()

    def get_events(self):
        events = super().get_events()
        if self.ind_res is not None:
            if self.phase_names[self.phase] == 'stim_' + str(self.ind_res):
                self.idx = self.session.global_log.shape[0]

        if len(events) > 0:
            self.count+=1
            for key, t in events:
                if self.idx is None:
                    pass
                    # self.session.beep.play() # Beep if response at no-response trial
                    print('Shoude be no response')
                elif self.idx is not None:
                    if self.count==1:
                        if key in self.parameters['correct_kb'] and self.parameters[
                            'correct_kb'].index(key)+1>=int((self.phase-2)/2):
                            self.session.res_feedback[self.trial_nr] = 1
                            print('Correct response')
                        else:
                            # self.session.beep.play()
                            print('Wrong response')
                            pass
                    else:
                        pass
                        # self.session.beep.play()
                        # print('Too many responses')
        else:
            pass
        
        if self.idx is not None:
            if getTime()-self.session.global_log.loc[self.idx-1, 'onset']>=2000:
                self.session.beep.play()
                print('Too long response')

    def run(self):

        #####################################################
        ## TRIGGER HERE
        #####################################################
        self.session.parallel_trigger(self.session.settings['design'].get('ttl_trigger_bar'))
        
        super().run()


class LocalizerTrial(Trial):
    """
    """
    def __init__(self, session, trial_nr, phase_durations, phase_names,
                 parameters, timing, verbose=True, draw_each_frame=False):
         
         super().__init__(session, trial_nr, phase_durations, phase_names,
                         parameters, timing, load_next_during_phase=None, verbose=verbose, draw_each_frame=draw_each_frame)

    def draw(self):
        self.session.fixation_w.draw()
        self.session.fixation_dot.draw()
        self.session.filled_circles.draw()
        if self.phase == 0:
            if self.parameters['category'] == 'object':
                self.session.fixation_cue[self.parameters['subcategory']].draw()
            elif self.parameters['category'] == 'face' or self.parameters['category'] == 'scene':
                self.session.image_stims['colored'][self.session.seq_subcate[self.parameters['trial_ind']]][self.session.seq_loc[self.parameters['trial_ind']]
                                ][self.session.seq_cate_ind[self.parameters['trial_ind']]].draw()
        if self.phase == 1:
            pass
        self.session.empty_circles.draw()
        self.session.win.flip()

    def run(self):
        #####################################################
        ## TRIGGER HERE
        #####################################################
        self.session.parallel_trigger(self.session.settings['design'].get('ttl_trigger_bar'))
        
        super().run()

class InstructionTrial(Trial):
    """ Simple trial with instruction text. """

    def __init__(self, session, trial_nr, phase_durations=[np.inf],
                 txt=None, keys=None, draw_each_frame=False, **kwargs):

        super().__init__(session, trial_nr, phase_durations, draw_each_frame=draw_each_frame, **kwargs)

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')
        text_position_x = self.session.settings['various'].get('text_position_x')
        text_position_y = self.session.settings['various'].get('text_position_y')

        if txt is None:
            txt = '''Press any button to continue.'''

        self.text = TextStim(self.session.win, txt,
                             height=txt_height, 
                             wrapWidth=txt_width, 
                             pos=[text_position_x, text_position_y],
                             font='Arial',
                             alignText = 'center',
                             anchorHoriz = 'center',
                             anchorVert = 'center')
        self.text.setSize(txt_height)

        self.keys = keys
        if hasattr(self.session, 'AT_image'):
            img0 = self.session.images_all[self.session.AT_image['category'][0]][self.session.AT_image['img_ind_in_pool'][0]]
            img1 = self.session.images_all[self.session.AT_image['category'][1]][self.session.AT_image['img_ind_in_pool'][1]]
            self.AT_image0 = ImageStim(win=self.session.win,
                                    image=img0,
                                    mask='circle',
                                    pos=[-1.4, -1.3],
                                    units='deg',
                                    name='AT_image',
                                    texRes=img0.shape[1],
                                    colorSpace='rgb',
                                    size=self.session.settings['stimuli'].get(
                                        'stim_size_deg'),
                                    interpolate=True)
            self.AT_image1 = ImageStim(win=self.session.win,
                                    image=img1,
                                    mask='circle',
                                    pos=[1.4, -1.3],
                                    units='deg',
                                    name='AT_image',
                                    texRes=img1.shape[1],
                                    colorSpace='rgb',
                                    size=self.session.settings['stimuli'].get(
                                        'stim_size_deg'),
                                    interpolate=True)

    def draw(self):
        self.session.fixation_w.draw()
        # self.session.report_fixation.draw()

        self.text.draw()
        if hasattr(self.session, 'AT_image'):
            self.AT_image0.draw()
            self.AT_image1.draw()
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
    """ Simple trial with text (trial x) and fixation. """

    def __init__(self, session, trial_nr, phase_durations=None,
                 txt="Waiting for scanner triggers.", draw_each_frame=False, **kwargs):

        super().__init__(session, trial_nr, phase_durations, txt, draw_each_frame=draw_each_frame, **kwargs)
    
    def draw(self):
        self.session.fixation_w.draw()
        if self.phase == 0:
            self.text.draw()
        else:
            # self.session.report_fixation.draw()
            pass
        self.session.win.flip()

    def get_events(self):
        events = Trial.get_events(self)

        if events:
            for key, t in events:
                if key == self.session.mri_trigger:
                    if self.phase == 0:
                        self.stop_phase()
                        self.session.win.flip()
                        #####################################################
                        ## TRIGGER HERE
                        #####################################################
                        self.session.experiment_start_time = getTime()
                        self.session.parallel_trigger(self.session.settings['design'].get('ttl_trigger_start'))

class OutroTrial(InstructionTrial):
    """ Simple trial with only fixation cross.  """

    def __init__(self, session, trial_nr, phase_durations, txt='', draw_each_frame=False, **kwargs):

        txt = ''''''
        super().__init__(session, trial_nr, phase_durations, txt=txt, draw_each_frame=draw_each_frame, **kwargs)

    def get_events(self):
        events = Trial.get_events(self)

        if events:
            for key, _ in events:
                if key == 'space':
                    self.stop_phase()   


class FeedbackTrial(Trial):
    """ Simple trial with instruction text. """

    def __init__(self, session, trial_nr, phase_durations=[np.inf],
                 txt=None, keys=None, draw_each_frame=False, **kwargs):

        super().__init__(session, trial_nr, phase_durations, draw_each_frame=draw_each_frame, **kwargs)
        self.keys = keys

    def draw(self):
        self.session.fixation_w.draw()
        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')
        text_position_x = self.session.settings['various'].get('text_position_x')
        text_position_y = self.session.settings['various'].get('text_position_y')

        if (self.session.res_feedback.count(1) + self.session.res_feedback.count(-1)) != 0:
            c = self.session.res_feedback.count(1) / (self.session.res_feedback.count(1) + self.session.res_feedback.count(-1))
            ACC = c*100
        else:
            ACC = 0

        txt_0 = f'Your score in this block is {str(math.ceil(ACC))}.'
        txt_1 = f'Please ask the experimenter to continue...'
        self.text_0 = TextStim(self.session.win, txt_0,
                             height=txt_height, 
                             wrapWidth=txt_width, 
                             pos=[text_position_x, text_position_y],
                             font='Arial',
                             alignText = 'center',
                             anchorHoriz = 'center',
                             anchorVert = 'center')
        self.text_1 = TextStim(self.session.win, txt_1,
                             height=txt_height, 
                             wrapWidth=txt_width, 
                             pos=[text_position_x, -text_position_y],
                             font='Arial',
                             alignText = 'center',
                             anchorHoriz = 'center',
                             anchorVert = 'center')
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