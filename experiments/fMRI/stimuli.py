#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from psychopy.core import getTime
from psychopy.visual import Line, Circle, GratingStim, TextStim
from psychopy.tools.colorspacetools import hsv2rgb


# from psychopy.filters import makeGrating
def makeGrating(res=256, color="white"):
    onePeriodX, onePeriodY = np.mgrid[
        0:res, 0 : 2 * np.pi : 1j * res
    ]
    grating = np.sin(onePeriodY - np.pi / 2)
    # initialise a 'black' texture
    color_grating = np.ones((res, res, 3)) * -1.0
    # replace the blue channel with the grating
    if color == "blue":
        color_grating[..., -1] = grating
    elif color == "red":
        color_grating[..., 0] = grating
    elif color == "green":
        color_grating[..., 1] = grating
    elif color == "white":
        color_grating[..., 0] = grating
        color_grating[..., 1] = grating
        color_grating[..., 2] = grating
    return color_grating


def makeGrating_hsv(res=256, color="white"):
    onePeriodX, onePeriodY = np.mgrid[
        0:res, 0 : 2 * np.pi : 1j * res
    ]
    grating = np.sin(onePeriodY - np.pi / 2)
    color_grating = np.ones((res, res, 3))
    if color == "red":
        color_grating[..., 0] = 0
    elif color == "green":
        color_grating[..., 0] = 120
    elif color == "blue":
        color_grating[..., 0] = 240
        color_grating[..., 1] = 1
    elif color == "white":
        color_grating[..., 0] = 0
        color_grating[..., 1] = 0
    
    color_grating[..., 2] = (grating+1)/2
    return hsv2rgb(color_grating)

    


def make_color_sqrXsqr_grating_tex(res=256, color="white"):
    res = 256
    color_grating = np.ones((res, res, 3)) * -1.0

    onePeriodX, onePeriodY = np.mgrid[
        0 : 2 * np.pi : 1j * res, 0 : 2 * np.pi : 1j * res
    ]
    sinusoid = np.sin(onePeriodX - np.pi / 2) * np.sin(onePeriodY - np.pi / 2)
    grating = np.where(sinusoid > 0, 1, -1)

    # replace the blue channel with the grating
    if color == "blue":
        color_grating[..., -1] = grating
    elif color == "red":
        color_grating[..., 0] = grating
    elif color == "green":
        color_grating[..., 1] = grating
    elif color == "white":
        color_grating[..., 0] = grating
        color_grating[..., 1] = grating
        color_grating[..., 2] = grating
    else:
        raise ValueError("color should be 'red', 'green', 'blue', or 'white'")
    return color_grating


def make_color_sqrXsqr_grating_hsvtex(res=256, color="white"):
    res = 256

    color_grating = np.ones((res, res, 3))
    if color == "red":
        color_grating[..., 0] = 0
        color_grating[..., 1] = 1
        color_grating[..., 2] = 1
    elif color == "green":
        color_grating[..., 0] = 120
        color_grating[..., 1] = 1
        color_grating[..., 2] = 1
    elif color == "blue":
        color_grating[..., 0] = 240
        color_grating[..., 1] = 1
        color_grating[..., 2] = 1
    elif color == "white":
        color_grating[..., 0] = 0
        color_grating[..., 1] = 0
        color_grating[..., 2] = 1

    onePeriodX, onePeriodY = np.mgrid[
        0 : 2 * np.pi : 1j * res, 0 : 2 * np.pi : 1j * res
    ]
    sinusoid = np.sin(onePeriodX - np.pi / 2) * np.sin(onePeriodY - np.pi / 2)
    grating = np.where(sinusoid > 0, 1, -1)
    
    color_grating[..., 2] = (grating+1)/2

    return hsv2rgb(color_grating)


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
        self,
        win,
        circle_radius,
        dotcolor,
        linecolor,
        contrast=1,
        cross_lindwidth=4,
        pos=[0, 0],
        edges=360,
        *args,
        **kwargs
    ):
        self.dotcolor = dotcolor
        self.linecolor = linecolor
        self.circle_radius = circle_radius
        self.pos = pos
        self.beta_outer_circle = 0.5
        self.beta_inner_circle = 0.2
        self.contrast = contrast
        self.cross_lindwidth = cross_lindwidth

        self.outer_circle = Circle(
            win,
            units="deg",
            radius=circle_radius * self.beta_outer_circle,
            pos=pos,
            edges=edges,
            fillColor=self.dotcolor,
            lineColor=self.dotcolor,
            contrast=contrast,
            *args,
            **kwargs
        )
        self.line1 = Line(
            win,
            units="deg",
            start=(-circle_radius * 0.55 + pos[0], pos[1]),
            end=(circle_radius * 0.55 + pos[0], pos[1]),
            contrast=contrast,
            lineColor=self.linecolor,
            lineWidth=self.cross_lindwidth,
        )
        self.line2 = Line(
            win,
            units="deg",
            start=(0, -circle_radius * 0.55 + pos[1]),
            end=(0, circle_radius * 0.55 + pos[1]),
            contrast=contrast,
            lineColor=self.linecolor,
            lineWidth=self.cross_lindwidth,
        )
        self.inner_circle = Circle(
            win,
            units="deg",
            radius=circle_radius * self.beta_inner_circle,
            pos=pos,
            edges=edges,
            fillColor=self.dotcolor,
            lineColor=self.dotcolor,
            contrast=contrast,
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
        self,
        win,
        freq,
        circle_radius,
        dotcolor,
        linecolor,
        contrast=1,
        cross_lindwidth=4,
        pos=[0, 0],
        edges=360,
        *args,
        **kwargs
    ):
        super().__init__(
            win,
            circle_radius,
            dotcolor,
            linecolor,
            contrast=contrast,
            cross_lindwidth=cross_lindwidth,
            pos=pos,
            edges=edges,
            *args,
            **kwargs
        )
        self.freq = freq
        self.inner_circle.opacity = 1.0
        self.outer_circle.opacity = 1.0
        # self.beta_inner_circle = 0.1
        # self.cross_lindwidth = cross_lindwidth/2

        self.inner_circle = Circle(
            win,
            units="deg",
            radius=circle_radius * self.beta_inner_circle,
            pos=pos,
            edges=edges,
            fillColor=self.dotcolor,
            lineColor=self.dotcolor,
            contrast=contrast,
            *args,
            **kwargs
        )
        self.last_time = getTime()

    def draw(self):
        present_time = getTime()
        if (present_time - self.last_time) > (1.0 / (self.freq * 2)):
            self.inner_circle.opacity = 1.0 - self.inner_circle.opacity
            self.outer_circle.opacity = 1.0 - self.outer_circle.opacity
            self.last_time = present_time
        self.outer_circle.draw()
        self.line1.draw()
        self.line2.draw()
        self.inner_circle.draw()


class Gabors(object):
    def __init__(
        self,
        win,
        size,
        sf,
        ori,
        color=(1.0, 1.0, 1.0),
        ecc=100,
        roll_dist=0,
        angle=0,
        phase=0,
        contrast=1,
        units="deg",
        *args,
        **kwargs
    ):

        grating_res = 256
        color_grating = makeGrating_hsv(res=grating_res, color=color)
        
        self.colorbg = GratingStim(
            win,
            tex=color_grating,
            mask="raisedCos",
            size=size,
            sf=sf,
            ori=ori,
            # color=color,
            colorSpace="rgb",
            pos=ecc * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
            + np.array([0, roll_dist]),
            phase=phase,
            contrast=contrast,
            units=units,
            *args,
            **kwargs
        )
        # self.gabor = GratingStim(
        #     win,
        #     tex="sin",
        #     mask="raisedCos",
        #     size=size,
        #     sf=sf,
        #     ori=ori,
        #     color=color,
        #     pos=ecc * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
        #     + np.array([0, roll_dist]),
        #     phase=phase,
        #     contrast=contrast,
        #     units=units,
        #     *args,
        #     **kwargs
        # )

    def draw(self):
        self.colorbg.draw()
        # self.gabor.draw()


class Checkerboards(object):
    def __init__(
        self,
        win,
        size,
        sf,
        ori,
        colorswap=False,
        color="white",
        ecc=100,
        roll_dist=0,
        angle=0,
        phase=0,
        contrast=1,
        units="deg",
        temporal_freq=8,
        *args,
        **kwargs
    ):
        """
        If colorswap is True, the checkerboards will swap between two colors.
        If colorswap is False, the checkerboards will swap between black and white.
        """
        self.temporal_freq = temporal_freq
        self.contrast = contrast
        self.colorswap = colorswap
        if self.colorswap:
            color_grating_1 = make_color_sqrXsqr_grating_hsvtex(color=color[0])
            color_grating_2 = make_color_sqrXsqr_grating_hsvtex(color=color[1])
            self.checkerboards_1 = GratingStim(
                win,
                tex=color_grating_1,  # "sqrXsqr",
                mask="raisedCos",
                size=size,
                sf=sf,
                ori=ori,
                pos=ecc
                * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
                + np.array([0, roll_dist]),
                phase=phase,
                contrast=contrast,
                units=units,
                *args,
                **kwargs
            )
            self.checkerboards_2 = GratingStim(
                win,
                tex=color_grating_2,  # "sqrXsqr",
                mask="raisedCos",
                size=size,
                sf=sf,
                ori=ori,
                pos=ecc
                * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
                + np.array([0, roll_dist]),
                phase=phase,
                contrast=contrast,
                units=units,
                *args,
                **kwargs
            )
            self.checkerboards = [self.checkerboards_1, self.checkerboards_2]
            self.checkerboard_index = int(0)
        else:
            color_grating = make_color_sqrXsqr_grating_tex(color=color)
            self.checkerboards = GratingStim(
                win,
                tex=color_grating,  # "sqrXsqr",
                mask="raisedCos",
                size=size,
                sf=sf,
                ori=ori,
                pos=ecc
                * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
                + np.array([0, roll_dist]),
                phase=phase,
                contrast=contrast,
                units=units,
                *args,
                **kwargs
            )
        self.last_time = getTime()

    def draw(self):
        present_time = getTime()
        if self.colorswap:
            if (present_time - self.last_time) > (1.0 / (self.temporal_freq * 2)):
                self.checkerboards[self.checkerboard_index].contrast = (
                    -self.checkerboards[self.checkerboard_index].contrast
                )
                self.checkerboards[self.checkerboard_index].ori += 45
                self.checkerboard_index = int(1 - self.checkerboard_index)
                self.last_time = present_time
            if (present_time - self.last_time) > (1.0 / (self.temporal_freq * 4)):
                self.checkerboards[self.checkerboard_index].draw()
        else:
            if (present_time - self.last_time) > (1.0 / (self.temporal_freq * 2)):
                self.checkerboards.contrast = -self.checkerboards.contrast
                self.checkerboards.ori += 45
                self.last_time = present_time
            if (present_time - self.last_time) > (1.0 / (self.temporal_freq * 4)):
                self.checkerboards.draw()


class CheckerboardsAdjContrast(Checkerboards):
    def __init__(
        self,
        win,
        size,
        sf,
        ori,
        colorswap=False,
        color="white",
        ecc=100,
        roll_dist=0,
        angle=0,
        direction=0,
        adj_rate=0,
        phase=0,
        contrast=1,
        units="deg",
        temporal_freq=8,
        *args,
        **kwargs
    ):
        super().__init__(
            win,
            size,
            sf,
            ori,
            colorswap,
            color,
            ecc,
            roll_dist,
            angle,
            phase,
            contrast,
            units,
            temporal_freq,
            *args,
            **kwargs
        )
        self.direction = direction
        if direction == 0:
            self.beta = 0
        elif direction == 1 or direction == -1:
            self.beta = adj_rate
        else:
            raise ValueError("direction should be 0, 1, or -1")
        self.last_time_contrast = getTime()
        self.inverse = 1

    def draw(self):
        present_time = getTime()
        if self.colorswap:
            if (present_time - self.last_time) > 0.1:
                self.last_time = present_time
                self.last_time_contrast = present_time
                self.checkerboards[self.checkerboard_index].contrast = self.contrast
                self.checkerboards[self.checkerboard_index].draw()
            else:
                self.inverse = -self.inverse
                if (present_time - self.last_time) > (1.0 / (self.temporal_freq * 2)):
                    self.checkerboards[self.checkerboard_index].contrast = (
                        self.inverse
                        * (
                            abs(self.checkerboards[self.checkerboard_index].contrast)
                            + self.beta
                            * (present_time - self.last_time_contrast)
                            * self.direction
                        )
                    )
                    self.checkerboards[self.checkerboard_index].ori += 45
                    self.checkerboard_index = int(1 - self.checkerboard_index)
                    self.last_time = present_time
                    self.last_time_contrast = present_time
                if (present_time - self.last_time) > (1.0 / (self.temporal_freq * 4)):
                    for i in range(2):
                        self.checkerboards[i].contrast = (
                            abs(self.checkerboards[i].contrast)
                            + self.beta
                            * (present_time - self.last_time_contrast)
                            * self.direction
                        )
                    self.last_time_contrast = present_time
                    self.checkerboards[self.checkerboard_index].draw()

        else:
            if (present_time - self.last_time) > 0.1:
                self.last_time = present_time
                self.last_time_contrast = present_time
                self.checkerboards.contrast = self.contrast
                self.checkerboards.draw()
            else:
                self.inverse = -self.inverse
                if (present_time - self.last_time) > (1.0 / (self.temporal_freq * 2)):
                    self.checkerboards.contrast = self.inverse * (
                        abs(self.checkerboards.contrast)
                        + self.beta
                        * (present_time - self.last_time_contrast)
                        * self.direction
                    )
                    self.checkerboards.ori += 45
                    self.last_time = present_time
                    self.last_time_contrast = present_time
                if (present_time - self.last_time) > (1.0 / (self.temporal_freq * 4)):
                    self.checkerboards.contrast = (
                        abs(self.checkerboards.contrast)
                        + self.beta
                        * (present_time - self.last_time_contrast)
                        * self.direction
                    )
                    self.last_time_contrast = present_time
                    self.checkerboards.draw()


class PlaceHolder(object):
    def __init__(
        self,
        win,
        circle_radius,
        color,
        ecc=100,
        roll_dist=0,
        angle=0,
        edges=360,
        linewidth=0.4,
        fill=False,
        *args,
        **kwargs
    ):
        self.color = color
        if fill:
            fill_color = "gray"
        else:
            fill_color = None
        self.circle = Circle(
            win,
            radius=circle_radius / 2,
            pos=ecc * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
            + np.array([0, roll_dist]),
            edges=edges,
            fillColor=fill_color,
            lineColor=color,
            lineWidth=linewidth,
            units="deg",
        )

    def draw(self):
        self.circle.draw()


class Highlighter(object):
    def __init__(
        self,
        win,
        txt,
        circle_radius,
        linecolor,
        ecc=100,
        roll_dist=0,
        angle=0,
        edges=360,
        linewidth=0.4,
        fillcolor=1,
        *args,
        **kwargs
    ):
        self.circle = Circle(
            win,
            radius=circle_radius / 2,
            pos=ecc * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
            + np.array([0, roll_dist]),
            edges=edges,
            fillColor=fillcolor,
            lineColor=linecolor,
            lineWidth=linewidth,
            units="deg",
        )
        self.questionmark = TextStim(
            win,
            text=txt,
            pos=ecc * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
            + np.array([0, roll_dist]),
            color=0.5,
            height=circle_radius / 1.4,
            units="deg",
        )

    def draw(self):
        self.circle.draw()
        self.questionmark.draw()


class Number(object):
    def __init__(
        self,
        win,
        circle_radius,
        ecc=100,
        roll_dist=0,
        angle=0,
        number=0,
        *args,
        **kwargs
    ):
        self.number = TextStim(
            win,
            text=str(number),
            pos=ecc * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
            + np.array([0, roll_dist]),
            color=0.5,
            height=circle_radius / 1.4,
            units="deg",
        )

    def draw(self):
        self.number.draw()
