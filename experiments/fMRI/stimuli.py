#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from psychopy.visual import Line, Circle, ImageStim, Window, GratingStim, RadialStim, PatchStim


class FixationBullsEye(object):
    def __init__(
        self, win, circle_radius, color, pos=[0, 0], edges=360, *args, **kwargs
    ):
        self.color = color
        # self.line1 = Line(
        #     win,
        #     start=(-circle_radius + pos[0], -circle_radius + pos[1]),
        #     end=(circle_radius + pos[0], circle_radius + pos[1]),
        #     lineColor=self.color,
        #     *args,
        #     **kwargs
        # )
        # self.line2 = Line(
        #     win,
        #     start=(-circle_radius + pos[0], circle_radius + pos[1]),
        #     end=(circle_radius + pos[0], -circle_radius + pos[1]),
        #     lineColor=self.color,
        #     *args,
        #     **kwargs
        # )
        self.circle1 = Circle(
            win,
            radius=circle_radius * 0.5,
            edges=edges,
            fillColor=None,
            lineColor=self.color,
            *args,
            **kwargs
        )
        self.circle2 = Circle(
            win,
            radius=circle_radius * 0.375,
            edges=edges,
            fillColor=None,
            lineColor=self.color,
            *args,
            **kwargs
        )
        self.circle3 = Circle(
            win,
            radius=circle_radius * 0.25,
            edges=edges,
            fillColor=None,
            lineColor=self.color,
            *args,
            **kwargs
        )
        self.circle4 = Circle(
            win,
            radius=circle_radius * 0.125,
            edges=edges,
            fillColor=None,
            lineColor=self.color,
            *args,
            **kwargs
        )

    def draw(self):
        # self.line1.draw()
        # self.line2.draw()
        self.circle1.draw()
        self.circle2.draw()
        self.circle3.draw()
        self.circle4.draw()

    def setColor(self, color):
        # self.line1.color = color
        # self.line2.color = color
        self.circle1.color = color
        self.circle2.color = color
        self.circle3.color = color
        self.circle4.color = color
        self.color = color


class FixationCue(object):
    def __init__(
        self, win, circle_radius, color, cross_lindwidth=4, pos=[0, 0], edges=360, *args, **kwargs
    ):
        self.color = color
        self.outer_circle = Circle(
            win,
            units="deg",
            radius=circle_radius * 0.5,
            pos=pos,
            edges=edges,
            fillColor=self.color,
            lineColor=self.color,
            *args,
            **kwargs
        )
        self.line1 = Line(
            win,
            units="deg",
            start=(-circle_radius + pos[0], 0),
            end=(circle_radius + pos[0], 0),
            lineColor=[0, 0, 0],
            lineWidth=cross_lindwidth,
        )
        self.line2 = Line(
            win,
            units="deg",
            start=(0, -circle_radius + pos[1]),
            end=(0, circle_radius + pos[1]),
            lineColor=[0, 0, 0],
            lineWidth=cross_lindwidth,
        )
        self.inner_circle = Circle(
            win,
            units="pix",
            radius=cross_lindwidth * 0.5,
            pos=pos,
            edges=edges,
            fillColor=self.color,
            lineColor=self.color,
            *args,
            **kwargs
        )

    def draw(self):
        self.outer_circle.draw()
        self.line1.draw()
        self.line2.draw()
        self.inner_circle.draw()


class PlaceHolderCircles(object):
    def __init__(
        self,
        win,
        circle_radius,
        color,
        ecc=100,
        edges=360,
        linewidth=0.4,
        fill=True,
        *args,
        **kwargs
    ):
        self.color = color
        if fill:
            fill_color = "gray"
        else:
            fill_color = None
        self.circles = [
            Circle(
                win,
                radius=circle_radius / 2,
                pos=ecc * np.array([np.sin(angle), np.cos(angle)]),
                edges=edges,
                fillColor=fill_color,
                lineColor=color,
                lineWidth=linewidth,
                units="deg",
            )
            for angle in np.pi / 4 + np.linspace(0, 2 * np.pi, 4, endpoint=False)
        ]

    def draw(self):
        for c in self.circles:
            c.draw()


class Gabors(object):
    # def __init__(
    #     self, win, tex, mask, *args, **kwargs
    # ):
    def __init__(
        self, win, size, sf, ori, ecc=100, angle=0, phase=0, contrast=1, units="deg", *args, **kwargs
    ):
        self.gabor = GratingStim(
            win,
            tex='sin',
            mask='raisedCos',
            size=size,
            sf=sf,
            ori=ori,
            pos=ecc * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))]),
            phase=phase,
            contrast=contrast,
            units=units,
            *args,
            **kwargs
        )

    def draw(self):
        self.gabor.draw()

class Checkerboards(object):
    # def __init__(
    #     self, win, tex, mask, *args, **kwargs
    # ):
    def __init__(
        self, win, size, sf, ori, ecc=100, angle=0, phase=0, contrast=1, units="deg", *args, **kwargs
    ):
        self.checkerboards = GratingStim(
            win,
            tex="sqrXsqr",
            mask="raisedCos",
            size=size,
            sf=sf,
            ori=ori,
            pos=ecc * np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))]),
            phase=phase,
            contrast=contrast,
            units=units,
            *args,
            **kwargs
        )

    def draw(self):
        self.checkerboards.draw()
