# Try to convert results to fsaverage and fsnative
in_file=/tank/shared/2023/dist_supp_pRF/pilot_data/derivatives/vonmises_pRF/vonmises-GLMsingle-lib-24loc/prf_best_kappa.nii.gz
fn_ext=.nii.gz
fn_dir=`dirname ${in_file}`
fn_base=`basename ${in_file} ${fn_ext}`
export SUBJECTS_DIR=/tank/shared/2023/dist_supp_pRF/pilot_data/derivatives/freesurfer/

conda info --envs
conda activate preproc

cd ${fn_dir}
warp=/tank/shared/tmp/suppr_pRF/fmriprep_wf/single_subject_002_wf/func_preproc_ses_01_task_ping_run_06_wf/bold_reg_wf/bbreg_wf/concat_xfm/out_fwd.tfm
call_antsapplytransforms --verbose ../../../derivatives/fmriprep/sub-002/ses-01/anat/sub-002_ses-01_acq-MP2RAGE_desc-preproc_T1w.nii.gz $in_file ${fn_base}_space-T1w.nii.gz $warp
warp2=`readlink -f ../../../derivatives/fmriprep/sub-002/ses-01/anat/sub-002_ses-01_acq-MP2RAGE_from-T1w_to-fsnative_mode-image_xfm.txt`
call_antsapplytransforms --verbose ../../../derivatives/freesurfer/sub-002/mri/orig.mgz $in_file ${fn_base}_space-fsnative.nii.gz $warp,$warp2
echo $warp
fslsize ${fn_base}_space-fsnative.nii.gz
readlink -f ../../../derivatives/freesurfer
echo $SUBJECTS_DIR
call_vol2fsaverage -o $PWD -p sub-002_ses-1 sub-002 ${fn_base}_space-fsnative.nii.gz desc-${fn_base}
