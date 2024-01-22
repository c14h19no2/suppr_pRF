import numpy as np
import pandas as pd
import nibabel as nib
import math
from os.path import join, exists
from pathlib import Path
import time
from bids.layout import BIDSLayout
import logging
from glmsingle.glmsingle import GLM_single


def save_nii(data, refdata, outputdir, filename):
    filepath = Path(outputdir, filename)
    nib.save(
        nib.Nifti1Image(
            data,
            affine=refdata.affine,
            header=refdata.header,
        ),
        filepath,
    )


def con_dm(ev_df, design_opt):
    TR_nr_from_DM = ev_df["event_type"].value_counts().pulse
    if TR_nr_from_DM + design_opt["blank_TR_nr"] != design_opt["TR_nr"]:
        logging.warning(
            f'Number of TRs from DM ({TR_nr_from_DM}) is not equal to the number of TRs from design_opt ({design_opt["TR_nr"]})'
        )
    start_timepoint = ev_df[
        (ev_df["trial_nr"] == 1) & (ev_df["event_type"] == "pulse")
    ]["onset"].values[0]
    # Correct the start time
    ev_df["onset"] = ev_df["onset"] - start_timepoint
    onset_df = ev_df[["onset", "angle_Ping", "ori_Ping", "direction"]][
        (ev_df["trial_type"] == "PingTrial") & (ev_df["event_type"] == "stimulus")
    ].reset_index(drop=True)

    onset_df["onset"] = onset_df["onset"] * 10
    onset_df["onset"] = onset_df["onset"].astype("int")
    onset_df["angle_Ping"] = onset_df["angle_Ping"].astype("int")
    onset_df["ori_Ping"] = onset_df["ori_Ping"].astype("int")

    dm = np.zeros(
        (
            len(np.arange(0, design_opt["total_time"], design_opt["pseudo_TR"])),
            len(design_opt["angles"]),
        ),
        dtype=int,
    )
    for _, row in onset_df.iterrows():
        TR_ind = int(
            (row["onset"] - design_opt["fixation_dur"] * 10)
            / (design_opt["pseudo_TR"] * 10)
        )
        column_ind = np.where(design_opt["angles"] == row["angle_Ping"])
        dm[TR_ind, column_ind] = 1

    return dm, onset_df


def con_dms(bids_layout, design_opt):
    """Design matrix"""
    sub = design_opt["sub"]
    ses = design_opt["ses"]
    runs = design_opt["runs"]
    ev_dfs = []
    dms = []
    stim_dms = []
    for run in runs:
        logging.info(f"Loading event file for run {run}")
        event_file = bids_layout.get(
            subject=sub,
            session=ses,
            task="ping",
            run=run,
            suffix="events",
            extension="tsv",
        )[0]
        logging.info(f"Event file path: {event_file.path}")
        ev_dfs.append(pd.read_csv(event_file.path, sep="\t"))

    for ev_df in ev_dfs:
        dm, onset_df = con_dm(ev_df, design_opt)
        dms.append(dm)
        stim_dms.append(np.array(onset_df["angle_Ping"]))
    return dms, stim_dms


def con_imgs(fmriprep_layout, design_opt):
    # nifti file
    imgs = []
    sub = design_opt["sub"]
    ses = design_opt["ses"]
    runs = design_opt["runs"]

    for run in runs:
        logging.info(f"Loading func image for run {run}")
        nifti_file = fmriprep_layout.get(
            subject=sub,
            session=ses,
            task="ping",
            run=run,
            space="T1w",
            suffix="bold",
            extension="nii.gz",
        )[0]
        logging.info(f"BOLD file path: {nifti_file.path}")
        datvol = nib.load(nifti_file)
        imgs.append(datvol.get_fdata())
        logging.info(f"The image size of run {run} is {imgs[-1].shape}")
    return imgs


def solve_offset(betas, dm_theta):
    # beta = a * cos(theta) * cos(offset) - a * sin(theta) * sin(offset) + b
    tanx_offset, _, _, _ = np.linalg.lstsq(dm_theta.T, betas)
    offset = math.degrees(math.atan2(tanx_offset[1], tanx_offset[0]))
    return offset


def create_offsetdm(degrees):
    radi = np.zeros((1, len(degrees)))
    dm_theta = np.zeros((2, len(degrees)))
    for count, degree in enumerate(degrees):
        radi[0, count] = math.radians(degree)
        dm_theta[0, count] = math.cos(radi[0, count])
        dm_theta[1, count] = math.sin(radi[0, count])
    return dm_theta


def cal_vertex_deg(betamap_all, unique_event_types):
    offset = np.zeros((1, betamap_all.shape[1]))
    for voxel_ind in range(betamap_all.shape[1]):
        betas = betamap_all[:, voxel_ind]
        dm_theta = create_offsetdm(unique_event_types)
        offset[0, voxel_ind] = solve_offset(betas, dm_theta)
    offset %= 360
    return offset


def fit_GLMsingle(
    design_opt,
    path_opt,
    GLMsingle_opt,
    output_typeC_retinamap=False,
    output_typeD_retinamap=False,
):
    datadir = path_opt["datadir"]
    datadir_bids = path_opt["datadir_bids"]
    derivatives = path_opt["derivatives"]
    datadir_fmriprep = path_opt["datadir_fmriprep"]
    datadir_freesufer = path_opt["datadir_freesufer"]
    outputdir = path_opt["outputdir"]
    figuredir = path_opt["figuredir"]

    logging.info(f"directory of dataset: {datadir}")
    logging.info(f"directory to save outputs: {outputdir}")

    bids_layout = BIDSLayout(datadir_bids, validate=False)
    fmriprep_layout = BIDSLayout(datadir_fmriprep, validate=False)
    logging.info(f"BIDS data path: {datadir_bids}")
    logging.info(f"fmriprep data path: {datadir_fmriprep}")
    """Set parameters
    """

    sub = design_opt["sub"]
    ses = design_opt["ses"]
    runs = design_opt["runs"]
    angles = design_opt["angles"]
    TR = design_opt["TR"]
    pseudo_TR = design_opt["pseudo_TR"]
    TR_nr = design_opt["TR_nr"]
    blank_TR_nr = design_opt["blank_TR_nr"]
    fixation_dur = design_opt["fixation_dur"]
    stim_dur = design_opt["stim_dur"]
    total_time = (TR_nr - blank_TR_nr) * TR

    """Load dms and images
    """
    dms, stim_dms = con_dms(bids_layout, design_opt)
    stim_con_dms = np.concatenate(stim_dms)
    imgs = con_imgs(fmriprep_layout, design_opt)

    # T1 file
    T1_file = fmriprep_layout.get(
        subject=sub, session=ses, suffix="T1w", extension="nii.gz"
    )[0]
    T1vol = nib.load(T1_file)

    """
    Do GLMsingle
    """
    if GLMsingle_opt["analysis"]["wantmaxpolydeg"]:
        GLMsingle_opt["analysis"].pop("wantmaxpolydeg")
    else:
        GLMsingle_opt["analysis"]["maxpolydeg"] = [[0, 1] for _ in imgs]
        GLMsingle_opt["analysis"].pop("wantmaxpolydeg")

    # running python GLMsingle involves creating a GLM_single object
    # and then running the procedure using the .fit() routine
    glmsingle_obj = GLM_single(GLMsingle_opt["analysis"])

    # visualize all the hyperparameters
    logging.info(f"{glmsingle_obj.params}")

    start_time = time.perf_counter()

    if not exists(outputdir):
        logging.info(f"running GLMsingle...")
        results_glmsingle = glmsingle_obj.fit(
            dms,
            imgs,
            fixation_dur + stim_dur,
            pseudo_TR,
            outputdir=str(outputdir),
            figuredir=str(figuredir),
        )
    else:
        logging.info(f"loading existing GLMsingle outputs from directory: {outputdir}")

        # load existing file outputs if they exist
        results_glmsingle = dict()
        results_glmsingle["typea"] = np.load(
            join(outputdir, "TYPEA_ONOFF.npy"), allow_pickle=True
        ).item()
        results_glmsingle["typeb"] = np.load(
            join(outputdir, "TYPEB_FITHRF.npy"), allow_pickle=True
        ).item()
        results_glmsingle["typec"] = np.load(
            join(outputdir, "TYPEC_FITHRF_GLMDENOISE.npy"), allow_pickle=True
        ).item()
        results_glmsingle["typed"] = np.load(
            join(outputdir, "TYPED_FITHRF_GLMDENOISE_RR.npy"), allow_pickle=True
        ).item()

    elapsed_time = time.perf_counter() - start_time

    logging.info(
        f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}"
    )

    """Save files into nifti format
    """
    # Load refernece file
    bm_image_fn = fmriprep_layout.get(
        subject=sub,
        session=ses,
        task="ping",
        run=runs[0],
        space="T1w",
        suffix="mask",
        extension="nii.gz",
    )[0]
    bg_image_fn = fmriprep_layout.get(
        subject=sub,
        session=ses,
        task="ping",
        run=runs[0],
        space="T1w",
        suffix="boldref",
        extension="nii.gz",
    )[0]
    logging.info(f"The reference image is {bm_image_fn.path}")
    bref = nib.load(bg_image_fn)
    bmask = nib.load(bm_image_fn).get_fdata().astype(bool)

    # save ONOFF R2 file
    R2_ONOFF_masked = np.zeros_like(results_glmsingle["typea"]["onoffR2"])
    R2_ONOFF_masked[bmask] = results_glmsingle["typea"]["onoffR2"][bmask]
    save_nii(R2_ONOFF_masked, bref, outputdir, "TYPEA_onoffR2.nii.gz")

    # save type-C R2
    R2_typec_masked = np.zeros_like(results_glmsingle["typec"]["R2"])
    R2_typec_masked[bmask] = results_glmsingle["typec"]["R2"][bmask]
    save_nii(R2_typec_masked, bref, outputdir, "TYPEC_R2.nii.gz")

    # save type-D beta value
    betasmd = results_glmsingle["typed"]["betasmd"]
    betasmd_masked = np.zeros_like(betasmd)
    betasmd_masked[bmask] = betasmd[bmask]
    save_nii(betasmd_masked, bref, outputdir, "TYPED_betasmd.nii.gz")

    # save type-D R2 file
    R2_typed_masked = np.zeros_like(results_glmsingle["typed"]["R2"])
    R2_typed_masked[bmask] = results_glmsingle["typed"]["R2"][bmask]
    save_nii(R2_typed_masked, bref, outputdir, "TYPED_R2.nii.gz")

    # save type-D fractal value
    FRACvalue_masked = np.zeros_like(results_glmsingle["typed"]["FRACvalue"])
    FRACvalue_masked[bmask] = results_glmsingle["typed"]["FRACvalue"][bmask]
    save_nii(FRACvalue_masked, bref, outputdir, "TYPED_FRACvalue.nii.gz")

    if output_typeC_retinamap == True:
        # calculate type-C retinamap
        betas_typec = results_glmsingle["typec"]["betasmd"]
        betamap_typec_all = np.zeros((len(angles), np.prod(betas_typec.shape[0:3])))

        for ind, angle in enumerate(angles):
            betamap_typec_all[ind, :] = np.nanmean(
                betas_typec[:, :, :, stim_con_dms == angle], axis=3
            ).flatten()

        betas_typec_all_trials = np.zeros(
            (betas_typec.shape[-1], np.prod(betas_typec.shape[0:3]))
        )
        for ind in range(betas_typec.shape[-1]):
            betas_typec_all_trials[ind, :] = np.nan_to_num(
                betas_typec[:, :, :, ind].flatten()
            )

        betamap_typec_all = np.nan_to_num(betamap_typec_all)
        offset_typec = cal_vertex_deg(betas_typec_all_trials, stim_con_dms)
        offset_typec = offset_typec.reshape(R2_typec_masked.shape).copy()

        offset_typec_masked = np.empty_like(offset_typec)
        offset_typec_masked[:] = np.nan
        offset_typec_masked[bmask] = offset_typec[bmask]
        save_nii(offset_typec_masked, bref, outputdir, "TYPEC_retinamap.nii.gz")

    if output_typeD_retinamap == True:
        # calculate type-D retinamap
        betas_typed = results_glmsingle["typed"]["betasmd"]
        betamap_typed_all = np.zeros((len(angles), np.prod(betas_typed.shape[0:3])))

        for ind, angle in enumerate(angles):
            betamap_typed_all[ind, :] = np.nanmean(
                betas_typed[:, :, :, stim_con_dms == angle], axis=3
            ).flatten()

        betas_typed_all_trials = np.zeros(
            (betas_typed.shape[-1], np.prod(betas_typed.shape[0:3]))
        )
        for ind in range(betas_typed.shape[-1]):
            betas_typed_all_trials[ind, :] = np.nan_to_num(
                betas_typed[:, :, :, ind].flatten()
            )

        betamap_typed_all = np.nan_to_num(betamap_typed_all)
        offset_typed = cal_vertex_deg(betas_typed_all_trials, stim_con_dms)
        offset_typed = offset_typed.reshape(R2_typed_masked.shape).copy()

        offset_typed_masked = np.empty_like(offset_typed)
        offset_typed_masked[:] = np.nan
        offset_typed_masked[bmask] = offset_typed[bmask]

        # save offset file
        save_nii(offset_typed_masked, bref, outputdir, "TYPED_retinamap.nii.gz")
