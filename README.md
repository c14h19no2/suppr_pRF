# Suppr_pRF

## About The Project

## Built With

This experiment is based on `exptools2` and `psychopy`.

## Design

The experiment consists of 3 learning periods: neutral learning, biased learning 1, and biased learning 2. The participant will see two raised-cosine patches, and need to respond to the orientation of the angled patch, while ignoring a distractor patch that is vertically or horizontally oriented. Except for the Task Trials, we also have Ping Trials, Resting Trials, and Sucker Trials in this experiment.

### Full Picture

This study consists of several phases, in the following order:

1. Practice (1.5 minutes)

2. Period 1:

- Training 1. Approximately 5 minutes

- Training 2. Approximately 2.5 minutes

- Test 1. Approximately 7.56 minutes

- Test 2. Approximately 7.56 minutes

- Test 3. Approximately 7.56 minutes

3. Period 2:

- Training 1. Approximately 5 minutes

- Training 2. Approximately 2.5 minutes

- Test 1. Approximately 7.56 minutes

- Test 2. Approximately 7.56 minutes

- Test 3. Approximately 7.56 minutes

4. Period 3:

- Training 1. Approximately 5 minutes

- Training 2. Approximately 2.5 minutes

- Test 1. Approximately 7.56 minutes

- Test 2. Approximately 7.56 minutes

- Test 3. Approximately 7.56 minutes

**Summary**

Training 1: 192 task trials

Training 2: 48 task trials + 48 ping trials

Test: 48 task trials + 48 ping trials + 12 resting trials + 18 sucker trials

### Trial Design

#### Training

**Task trial**

Phase: Flickering fixation -> Stim (2 raised cosine patches, 4 possible locations) -> response time (black fixation) -> feedback (green or red fixation)

Time: 400 ms -> 400 ms -> 1000 ms -> 200 ms

**Ping trial**

Phase: Flickering fixation -> Ping (one spinning and flickering checkboard, 24 possible locations) -> response time -> feedback (green or red fixation)

Time: 400 ms -> 400 ms -> until response for maximum 1000 ms

#### Test

**Task trial**

- Phase: Flickering fixation -> Stim (2 raised cosine patches, 4 possible locations) -> response time

- Time: 400 ms -> 400 ms -> 2800 ms

**Ping trial**

- Phase: Flickering fixation -> Stim (one spinning and flickering checkboard, 24 possible locations) -> response time

- Time: 400 ms -> 400 ms -> 2800 ms

**Resting trial**

- Phase: Still fixation -> response time

- Time: 400 ms -> 3200 ms

**Sucker trial**

- Phase: Flickering fixation -> response time

- Time: 400 ms -> 3200 ms

## Usage

1. Go to <https://github.com/VU-Cog-Sci/exptools2> and install **exptools2**.
2. Activate exptools2:

    ```sh
    conda activate exptools2
    ```

3. Adjust the stimuli window:

    ```sh
    python roll_down_the_window.py $subject$ $eyetracking$
    ```

    This command is used to adjust the location of stimuli on monitor to make sure that participant\'s eyesight is not blocked and can see the full scene of the experiment.

4. Run the experiment:

    ```sh
    python main.py $subject$ $session$ $task$ $run$ $eyetracking$
    ```

Parameters:

- $subject$: The subject ID of the experiment, as a zero-filled integer, such as 01, or 04.
- $session$: The current session of the experiment, as a string, can be 'practice', 'train' or 'test'.
- $task$: The current task of the experiment, as a string, can be 'neutral', 'bias1', or 'bias2'; or an int, can be '0', '1', or '2'.
- $run$: The current run number of the experiment, an integer, such as 1, or 99.
- $eyetracking$: Switch the eyelink on and off, as an int, can be '0' or '1'.

## Some references

Fixdot used in this design: <https://www.sciencedirect.com/science/article/pii/S0042698912003380>

