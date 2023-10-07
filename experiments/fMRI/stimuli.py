#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from psychopy.core import getTime
from psychopy.visual import Line, Circle, GratingStim


class FixationBullsEye(object):
    def __init__(
        self, win, circle_radius, color, pos=[0, 0], edges=360, *args, **kwargs
    ):
        self.color = color
        self.circle1 = Circle(
            win,
            units="deg",
            pos=pos,
            radius=circle_radius * 0.65,
            edges=edges,
            fillColor=None,
            lineColor=self.color,
            *args,
            **kwargs
        )
        self.circle2 = Circle(
            win,
            units="deg",
            pos=pos,
            radius=circle_radius * 1.35,
            edges=edges,
            fillColor=None,
            lineColor=self.color,
            *args,
            **kwargs
        )

    def draw(self):
        self.circle1.draw()
        self.circle2.draw()

    def setColor(self, color):
        self.circle1.color = color
        self.circle2.color = color
        self.color = color


class FixationDot(object):
    def __init__(
        self, win, circle_radius, dotcolor, linecolor, cross_lindwidth=4, pos=[0, 0], edges=360, *args, **kwargs
    ):
        self.dotcolor = dotcolor
        self.linecolor = linecolor
        self.circle_radius = circle_radius
        self.pos = pos
        self.beta_outer_circle = 0.5
        self.beta_inner_circle = 0.2
        self.outer_circle = Circle(
            win,
            units="deg",
            radius=circle_radius * self.beta_outer_circle,
            pos=pos,
            edges=edges,
            fillColor=self.dotcolor,
            lineColor=self.dotcolor,
            *args,
            **kwargs
        )
        self.line1 = Line(
            win,
            units="deg",
            start=(-circle_radius*0.5 + pos[0], pos[1]),
            end=(circle_radius*0.5 + pos[0], pos[1]),
            lineColor=self.linecolor,
            lineWidth=cross_lindwidth,
        )
        self.line2 = Line(
            win,
            units="deg",
            start=(0, -circle_radius*0.5 + pos[1]),
            end=(0, circle_radius*0.5 + pos[1]),
            lineColor=self.linecolor,
            lineWidth=cross_lindwidth,
        )
        self.inner_circle = Circle(
            win,
            units="deg",
            radius=circle_radius * self.beta_inner_circle,
            pos=pos,
            edges=edges,
            fillColor=self.dotcolor,
            lineColor=self.dotcolor,
            *args,
            **kwargs
        )

    def draw(self):
        self.outer_circle.draw()
        self.line1.draw()
        self.line2.draw()
        self.inner_circle.draw()

class FixationDot_flk(FixationDot):
    def __init__(
        self, win, freq, circle_radius, dotcolor, linecolor, cross_lindwidth=4, pos=[0, 0], edges=360, *args, **kwargs
    ):
      super().__init__(win, circle_radius, dotcolor, linecolor, cross_lindwidth, pos, edges, *args, **kwargs)
      self.freq = freq
      self.inner_circle.opacity = 1.0
      self.outer_circle.opacity = 1.0
      self.beta_inner_circle = 0.1

      self.inner_circle = Circle(
            win,
            units="deg",
            radius=circle_radius * self.beta_inner_circle,
            pos=pos,
            edges=edges,
            fillColor=self.dotcolor,
            lineColor=self.dotcolor,
            *args,
            **kwargs
        )
      
      self.last_time = getTime()

    def draw(self):
        present_time = getTime()
        if (present_time - self.last_time) > (1.0/(self.freq * 2)):
            self.inner_circle.opacity = 1.0 - self.inner_circle.opacity
            self.outer_circle.opacity = 1.0 - self.outer_circle.opacity
            self.last_time = present_time
        self.outer_circle.draw()
        self.line1.draw()
        self.line2.draw()
        self.inner_circle.draw()

class Gabors(object):
    def __init__(
        self, win, size, sf, ori, ecc=100, roll_dist=0, angle=0, phase=0, contrast=1, units="deg", *args, **kwargs
    ):
        self.gabor = GratingStim(
            win,
            tex='sin',
            mask='raisedCos',
            size=size,
            sf=sf,
            ori=ori,
            pos=ecc * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))]) + np.array([0, roll_dist]),
            phase=phase,
            contrast=contrast,
            units=units,
            *args,
            **kwargs
        )

    def draw(self):
        self.gabor.draw()


class Checkerboards(object):
    def __init__(
        self, win, size, sf, ori, ecc=100, roll_dist=0, angle=0, phase=0, contrast=1, units="deg", temporal_freq=8, *args, **kwargs
    ):
        self.temporal_freq = temporal_freq
        self.contrast = contrast
        self.checkerboards = GratingStim(
            win,
            tex="sqrXsqr",
            mask="raisedCos",
            size=size,
            sf=sf,
            ori=ori,
            pos=ecc * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])+ np.array([0, roll_dist]),
            phase=phase,
            contrast=contrast,
            units=units,
            *args,
            **kwargs
        )
        self.last_time = getTime()

    def draw(self):
        present_time = getTime()
        if (present_time - self.last_time) > (1.0/(self.temporal_freq * 2)):
            self.checkerboards.contrast = -self.checkerboards.contrast
            self.checkerboards.ori += 45
            self.last_time = present_time
        if (present_time - self.last_time) > (1.0/(self.temporal_freq * 4)):
            self.checkerboards.draw()

class CheckerboardsAdjContrast(Checkerboards):
    def __init__(
        self, win, size, sf, ori, ecc=100, roll_dist=0, angle=0, direction=0, phase=0, contrast=1, units="deg", temporal_freq=8, *args, **kwargs
    ):
        super().__init__(win, size, sf, ori, ecc, roll_dist, angle, phase, contrast, units, temporal_freq, *args, **kwargs)
        self.direction = direction
        if direction == 0:
            self.beta = 0.5
        elif direction == 1:
            self.beta = 1.5
        elif direction == -1:
            self.beta = 0.4
        else:
            raise ValueError("direction should be 0, 1, or -1")
    
    def draw(self):
        present_time = getTime()
        if (present_time - self.last_time) > 0.1:
            self.last_time = present_time
            self.checkerboards.contrast = self.contrast
        if (present_time - self.last_time) > (1.0/(self.temporal_freq * 2)):
            self.checkerboards.contrast = self.checkerboards.contrast + self.beta * (present_time - self.last_time) * self.direction
            self.checkerboards.ori += 45
            self.last_time = present_time
        if (present_time - self.last_time) > (1.0/(self.temporal_freq * 4)):
            self.checkerboards.draw()