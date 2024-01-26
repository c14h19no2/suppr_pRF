import os
import time
import argparse
import scipy as sp
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging
import yaml
from bids.layout import BIDSLayout
import nibabel as nib
from prfpy.rf import vonMises1D as vm
from prfpy.rf import gauss1D_cart as g1d
from scipy.ndimage import median_filter, gaussian_filter, binary_propagation
from nilearn.glm.first_level.hemodynamic_models import _gamma_difference_hrf


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


def calc_stim_radius(stim_size, stim_center_dist):
    """Calculate the size of the circle in angle units from visual angle units"""
    return np.degrees(np.arctan((stim_size / 2) / stim_center_dist))


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


def con_vonmises_grid(
    angle_nr,
    oversamplingratio=9,
    kappas_nr=20,
    kappa_range=[0.1, 50],
):
    """
    :param angle_nr: number of angles
    :param oversamplingratio: oversampling ratio for mu
    :param kappa_nr: number of kappa values
    :param kappa_range: range of kappa values
    :return: 2D array of von mises values"""
    kappa_log10_range = [np.log10(kappa_range[0]), np.log10(kappa_range[1])]
    mus = np.linspace(0, 2 * np.pi, oversamplingratio * angle_nr, endpoint=False)
    kappas = np.logspace(
        kappa_log10_range[0], [kappa_log10_range[1]], kappas_nr, endpoint=True
    )
    mugrid, kappagrid = np.meshgrid(mus, kappas)
    mugrid, kappagrid = mugrid.ravel(), kappagrid.ravel()
    return (
        np.array([vm(mus, mu, kappa) for mu, kappa in zip(mugrid, kappagrid)]),
        mugrid,
        kappagrid,
    )


def angle_to_stim_screen(stim_angle, angle_nr, stim_radius, oversamplingratio=9):
    radians = np.linspace(
        0, np.radians(360), angle_nr * oversamplingratio, endpoint=False
    )
    low_bound = np.radians((stim_angle - stim_radius) % 360)
    high_bound = np.radians((stim_angle + stim_radius) % 360)
    stim_radian = np.radians(stim_angle % 360)
    idx_low_bound = (np.abs(radians - low_bound)).argmin()
    idx_high_bound = (np.abs(radians - high_bound)).argmin()
    stim_screen = np.zeros_like(radians)
    if (high_bound < low_bound) and (
        stim_radius >= 360 / (angle_nr * oversamplingratio * 2)
    ):
        stim_screen[idx_low_bound:] = 1
        stim_screen[: idx_high_bound + 1] = 1
    elif (high_bound < low_bound) and (
        stim_radius < 360 / (angle_nr * oversamplingratio * 2)
    ):
        stim_screen[: idx_high_bound + 1] = 1
    else:
        stim_screen[idx_low_bound : idx_high_bound + 1] = 1
    return stim_screen


def con_vonmises_dm(
    dm: np.ndarray,
    angle_nr: int = 8,
    stim_radius: int = 0.7,
    oversamplingratio: int = 9,
):
    angles = np.linspace(0, 360, angle_nr, endpoint=False)

    new_dm = np.zeros((dm.shape[0], angle_nr * oversamplingratio))
    for TR_ind in range(dm.shape[0]):
        if any(dm[TR_ind, :]):
            angle = angles[np.where(dm[TR_ind, :])[0][0]]
            new_dm[TR_ind, :] = angle_to_stim_screen(
                angle, angle_nr, stim_radius, oversamplingratio=oversamplingratio
            )
    return new_dm


def model_timecourse(model, dm):
    return np.dot(model.ravel(), dm.T.reshape((-1, dm.shape[0])))


def rsq_for_model(data, model_tcs):
    """
    Parameters
    ----------
    data : numpy.ndarray
        1D or 2D, containing single time-course or multiple
    model_tcs : numpy.ndarray
        1D, containing single model time-course
    Returns
    -------
    rsq : float or numpy.ndarray
        within-set rsq for this model's GLM fit, for all voxels in the data
    yhat : numpy.ndarray
        1D or 2D, model time-course for all voxels in the data

    """
    dm = np.array([np.ones(data.shape[-1]), model_tcs]).T
    betas, _, _, _ = np.linalg.lstsq(dm, data.T, rcond=None)
    yhat = np.dot(dm, betas).T
    rsq = 1 - (data - yhat).var(-1) / data.var(-1)
    return np.vstack((betas, rsq))


def grid_search_for_voxel(sv_tcs, grid_model_timecourses_conv, mugrid, kappagrid):
    b_rsqs = np.array(
        [rsq_for_model(sv_tcs, mtcs) for mtcs in grid_model_timecourses_conv]
    )

    max_rsq_ind = np.argmax(b_rsqs[:, -1, :], 0)
    best_rsq = np.array([b_rsqs[m, :, i] for i, m in enumerate(max_rsq_ind)])
    best_angle = np.array([mugrid[m] for _, m in enumerate(max_rsq_ind)])
    best_kappa = np.array([kappagrid[m] for _, m in enumerate(max_rsq_ind)])
    return np.vstack(
        (best_rsq.T, best_angle[np.newaxis, :], best_kappa[np.newaxis, :]),
    )


def main():
    """Parse arguments
    input: yml config file
    """

    parser = argparse.ArgumentParser(description="GLMsingle setup")
    parser.add_argument(
        "yml_config", default=None, nargs="?", help="yml config file path"
    )

    cmd_args = parser.parse_args()
    yml_config = cmd_args.yml_config

    # set up logging
    logging.basicConfig(
        filename=f"vonmises_pRF_logfile_{yml_config}.log",
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    """Exp parameters setup
    """
    if os.path.isfile(yml_config):
        with open(yml_config, "r") as ymlfile:
            try:
                opt = yaml.safe_load(ymlfile)
            except yaml.YAMLError as exc:
                logging.error(exc)
    else:
        raise ValueError(f"YAML file {yml_config} not found")

    design_opt = opt["EXP_opt"]["design"]
    path_opt = opt["EXP_opt"]["path"]
    vonmises_opt = opt["vonmises_GLMsingle_opt"]

    bids_data = path_opt["bids"]
    derivatives = path_opt["derivatives"]
    outputfolder = vonmises_opt["path"]["outputfolder"]

    # set exp parameters
    design_opt["runs"] = np.array(design_opt["runs"])
    design_opt["angles"] = np.array(
        [*range(0, 360, int(360 / design_opt["angles_nr"]))]
    )
    TR = design_opt["pseudo_TR"]
    design_opt["total_time"] = (
        design_opt["TR_nr"] - design_opt["blank_TR_nr"]
    ) * design_opt["TR"]

    # set path
    path_opt["datadir"] = Path(path_opt["datadir"])
    path_opt["datadir_bids"] = Path(path_opt["datadir"], bids_data)
    path_opt["datadir_fmriprep"] = Path(path_opt["datadir"], derivatives, "fmriprep")
    path_opt["datadir_freesufer"] = Path(path_opt["datadir"], derivatives, "freesurfer")
    path_opt["outputdir"] = Path(path_opt["datadir"], derivatives, outputfolder)
    path_opt["outputdir_task"] = Path(
        path_opt["datadir"], derivatives, outputfolder, vonmises_opt["path"]["name"]
    )
    path_opt["GLMsingle"] = Path(
        path_opt["datadir"],
        derivatives,
        "GLMsingle",
        opt["GLMsingle_opt"]["path"]["name"],
        opt["GLMsingle_opt"]["path"]["outputfolder"],
    )
    os.makedirs(path_opt["outputdir"], exist_ok=True)
    os.makedirs(path_opt["outputdir_task"], exist_ok=True)

    # set up BIDS layout
    bids_layout = BIDSLayout(path_opt["datadir_bids"], validate=False)
    fmriprep_layout = BIDSLayout(path_opt["datadir_fmriprep"], validate=False)

    # load reference image
    bg_image_fn = fmriprep_layout.get(
        subject=design_opt["sub"],
        session=design_opt["ses"],
        task="ping",
        run=design_opt["runs"][0],
        space="T1w",
        suffix="boldref",
        extension="nii.gz",
    )[0]
    bref = nib.load(bg_image_fn)

    # set up design matrix
    print(f"Design matrix setup started at {time.ctime()}")
    # load npy file
    _, stim_dms = con_dms(bids_layout, design_opt)
    stim_dm = np.concatenate(stim_dms, axis=0)
    dm = np.zeros((stim_dm.shape[0], design_opt["angles_nr"]))
    for TR_ind in range(stim_dm.shape[0]):
        stim_dm_trial_angle = np.argmin(abs(stim_dm[TR_ind] - design_opt["angles"]))
        dm[TR_ind, stim_dm_trial_angle] = 1
    oversamplingratio = vonmises_opt["oversamplingratio"]
    stim_radius = calc_stim_radius(0.7, 2)
    new_dm = con_vonmises_dm(
        dm,
        angle_nr=design_opt["angles_nr"],
        stim_radius=stim_radius,
        oversamplingratio=oversamplingratio,
    )

    (
        models,
        mugrid,
        kappagrid,
    ) = con_vonmises_grid(
        angle_nr=design_opt["angles_nr"],
        oversamplingratio=oversamplingratio,
        kappas_nr=20,
        kappa_range=[0.1, 50],
    )

    grid_model_timecourses = np.array(
        [model_timecourse(models[i, :], new_dm) for i in range(models.shape[0])]
    )

    # load image
    print(f"Loading images started at {time.ctime()}")
    TYPED_FITHRF_GLMDENOISE_RR = np.load(
        path_opt["GLMsingle"] / "TYPED_FITHRF_GLMDENOISE_RR.npy", allow_pickle=True
    ).item()
    img = np.array(TYPED_FITHRF_GLMDENOISE_RR["betasmd"])
    img_rsq = np.zeros(img.shape[:-1])
    img_best_angle = np.zeros(img.shape[:-1])
    img_best_kappa = np.zeros(img.shape[:-1])

    # Prepare the data
    img_2D = np.reshape(img, (np.prod(img.shape[0:-1]), img.shape[-1]))
    beta_whole_brain = np.zeros((np.prod(img.shape[0:-1])))
    rsq_whole_brain = np.zeros((np.prod(img.shape[0:-1])))
    best_angle_whole_brain = np.zeros((np.prod(img.shape[0:-1])))
    best_kappa_whole_brain = np.zeros((np.prod(img.shape[0:-1])))
    print(f"Reshaped image shape: {img_2D.shape}")

    start_time = time.perf_counter()
    print(f"pRF mapping started at {time.ctime()}")
    block_size = 100
    block_log_freq = 20
    block_nr = int(np.ceil(np.prod(img.shape[0:-1]) / block_size))
    for block in range(block_nr):
        sv_tcs = img_2D[block * block_size : (block + 1) * block_size, :]
        beta_rsq_angle_kappa = grid_search_for_voxel(
            np.array(sv_tcs),
            np.array(grid_model_timecourses),
            np.array(mugrid),
            np.array(kappagrid),
        )
        """beta_rsq_angle_kappa:
            [0, :] - beta 0
            [1, :] - beta 1
            [2, :] - best rsq
            [3, :] - best angle
            [4, :] - best kappa"""
        beta_whole_brain[
            block * block_size : (block + 1) * block_size
        ] = beta_rsq_angle_kappa[1, :]
        rsq_whole_brain[
            block * block_size : (block + 1) * block_size
        ] = beta_rsq_angle_kappa[2, :]
        best_angle_whole_brain[
            block * block_size : (block + 1) * block_size
        ] = beta_rsq_angle_kappa[3, :]
        best_kappa_whole_brain[
            block * block_size : (block + 1) * block_size
        ] = beta_rsq_angle_kappa[4, :]
        if (block == 0) or ((block + 1) % block_log_freq == 0):
            print(f"Processed {block + 1}/{block_nr} blocks")
            print(f"Time elapsed: {time.perf_counter()-start_time:.2f}s")
            avarage_time_per_block = (time.perf_counter() - start_time) / (block + 1)
            print(
                f"Avarage time per {block_log_freq} blocks: {avarage_time_per_block * block_log_freq:.2f}s"
            )
            print(
                f"Estimated time remaining: {(block_nr-block)*avarage_time_per_block/60:.2f}min"
            )
            print("-" * 25)

    img_betas = np.reshape(beta_whole_brain, img.shape[:-1])
    img_rsq = np.reshape(rsq_whole_brain, img.shape[:-1])
    img_best_angle = np.reshape(best_angle_whole_brain, img.shape[:-1])
    img_best_angle = np.degrees(img_best_angle)
    img_best_kappa = np.reshape(best_kappa_whole_brain, img.shape[:-1])

    # save result images
    save_nii(img_betas, bref, path_opt["outputdir_task"], "prf_betas.nii.gz")
    save_nii(img_rsq, bref, path_opt["outputdir_task"], "prf_rsq.nii.gz")
    save_nii(
        img_best_angle,
        bref,
        path_opt["outputdir_task"],
        "prf_best_angle.nii.gz",
    )
    save_nii(
        img_best_kappa,
        bref,
        path_opt["outputdir_task"],
        "prf_best_kappa.nii.gz",
    )


if __name__ == "__main__":
    main()
