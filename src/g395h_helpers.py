import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
from astropy.io import fits
from scipy.ndimage import median_filter as MF
from scipy.stats import median_abs_deviation as mad
import pickle
import pdb
from tqdm import tqdm
from ipywidgets import interact, widgets, HBox, Label, interactive
from jwst.pipeline import calwebb_detector1
from jwst import datamodels
from jwst.stpipe import Step
import logging

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if logger.name.startswith("jwst") or logger.name.startswith("stpipe"):
        logger.disabled = True

def get_frame(data_model, oversampling_factor = 10, row_min = 600, row_max = 2040):
    """
    Get frame ready to extract trace by taking the median over all groups and integrations

    Args:
        data_model: jwst model data with shape (nints, ngroups, nrows, ncols) after the jump step
    Returns:
        frame (np.ndarray): data array with shape (2, nrows, ncols) where the first dimension is the science ['SCI'] and error ['ERR'] frames
    """

    nints = data_model.data.shape[0]
    ngroups = data_model.data.shape[1]

    # Take median across all groups and integrations for the science and error frames
    flux_med = np.median([data_model.data[i][ngroups-1] for i in range(nints)],axis=0)
    err_med = np.median([data_model.err[i][ngroups-1] for i in range(nints)],axis=0)

    # Combine the science and error frames
    frame = np.array([flux_med, err_med])
    if np.any(~np.isfinite(frame[0])):
                    frame[0][~np.isfinite(frame[0])] = np.nan   

    # Flip the axes
    frame = np.array([np.flip(frame[0].T,axis=1),np.flip(frame[1].T,axis=1)])

    # Truncate to desired row min and row max
    frame = np.array([frame[0][row_min:row_max].astype(float),frame[1][row_min:row_max].astype(float)])
    
    # Oversample
    frame = np.array([resample_frame(frame[0],oversampling_factor),resample_frame(frame[1],oversampling_factor)])
    
    return frame


def resample_frame(data,oversampling=10,xmin=0,verbose=False):
    """A function that resamples all rows within an image to a greater sampling via linear interpolation. This is being tested as a method to deal with partial pixel extraction

    Inputs:
    data - the 2D spectral image
    oversampling - the number of sub-pixels in which to split each larger pixel into. Default=10
    xmin - if the data frame is a cut out of the larger frame, can define xmin as the left hand column for consistent x arrays. Default=0.
    verbose - True/False - do we want to plot the output of the resampling?

    Returns:
    data_resampled - the resampled image data"""

    nrows,ncols = data.shape
    old_x = np.arange(xmin,ncols)
    # new_x = np.arange(xmin,ncols-1+1/oversampling,1/oversampling)
    new_x = np.linspace(xmin,ncols,ncols*oversampling)

    data_resampled = np.array([np.interp(new_x,old_x,y) for y in data])

    return data_resampled


def find_spectral_trace(frame,guess_location,search_width,gaussian_width,trace_poly_order, oversampling_factor=10):
    """The function used to extract the location of a spectral trace either with a Gaussian or the argmax and then
    fits a nth order polynomial to these locations

    Args:
        frame:  3D image of shape (2, ncols, nrows) e.g. (2, 2048, 32) where the first dimension is the science frame [0] and the error frame [1]
        search_width (float): the search width (in pixels) around 'trace_guess_locations' where the code will try to locate the star. For crowded fields, set this to a narrower value. For poor pointing (lost of drift) set this to a wider value. 
        guess_locations (float): approximate location of the centre of the star's trace in x-pixel coordinates (in the cross-dispersion axis, can be found via DS9). If more than one star is to be extracted, separate the guess locations with a comma ","
        gaussian_width (float): width of the Gaussian that is used to fit the trace location along each row in the cross-dispersion axis. This value doesn't matter too much and so can be kept at the default value of 5.
        trace_poly_order (float): Define the order of the polynomial used to fit the centroids of the Gaussians in the dispersion axis. i.e., what order polynomial best describes the curvature of the spectral trace? In most cases, the default of 4 is sufficient. If using a spline, this must be set to 0.

    """

        
    # for JWST we need to extract only the first array since the frame is an array of (flux_frame,error_frame)
    frame = frame[0]

    buffer_pixels = 5 # ignore edges of detector which might have abnormally high counts

    if oversampling_factor:
        search_width = search_width * oversampling_factor
        guess_location = guess_location * oversampling_factor
        gaussian_width = gaussian_width * oversampling_factor

    if guess_location-search_width < buffer_pixels:
        search_left_edge = buffer_pixels
    else:
        search_left_edge = guess_location-search_width

    if guess_location+search_width > np.shape(frame)[1] - buffer_pixels:
        search_right_edge = np.shape(frame)[1] - buffer_pixels
    else:
        search_right_edge = guess_location+search_width

    columns_of_interest = frame[:,search_left_edge:search_right_edge]
    nrows,ncols = np.shape(columns_of_interest)

    trace_centre = []
    fwhm = []
    gauss_std = [] # the standard deviation in pixels measured by the Gaussian

    row_array = np.arange(nrows)

    plot_row = nrows//2


    for i, row in enumerate(columns_of_interest):


        x = np.arange(ncols)[np.isfinite(row)]+search_left_edge
        row = row[np.isfinite(row)]

        # pdb.set_trace()
        centre_guess = peak_counts_location = x[np.argmax(row)]
        amplitude = np.nanmax(row)
        amplitude_offset = np.nanmin(row)

        try:
            popt1,pcov1 = optimize.curve_fit(gauss,x,row,p0=[amplitude,centre_guess,gaussian_width,amplitude_offset])

            # Make sure fitted amplitude (with offset) is not less than 25% of the guess amplitude. - note for ACAM this number was 0.3 (70%).
            # print(search_left_edge,search_right_edge,popt1[1])
            if np.fabs(popt1[0] + popt1[-1] - amplitude) < amplitude * 0.75:
                TC = popt1[1] # trace centre
                trace_centre.append(TC)
                fwhm.append(popt1[2]*2*np.sqrt(2*np.log(2))) ### save width of gaussian as FWHM after applying conversion
                gauss_std.append(abs(popt1[2]))


            else:
                TC = centre_guess
                trace_centre.append(TC)
                nerrors += 1
                gauss_std.append(0) # append 0, this will be replaced by the mean of surrounding rows in extract_trace_flux
                fwhm.append(np.nan)


        except:
            TC = centre_guess
            trace_centre.append(TC)
            gauss_std.append(0) # append 0, this will be replaced by the mean of surrounding rows in extract_trace_flux
            fwhm.append(np.nan)




    # use a running median to smooth the centres, with a running box of 5 data points, before fitting with a polynomial
    trace_median_filter = MF(trace_centre,5)

    if trace_poly_order > 0: # we're using the user-defined polynomial order
        poly = np.poly1d(np.polyfit(row_array,trace_median_filter,trace_poly_order))
    else: # we use a polynomial of fourth order to find the outliers before fitting the spline (which may otherwise fit the outliers)
        poly = np.poly1d(np.polyfit(row_array,trace_median_filter,4))

    fitted_positions = poly(np.arange(nrows))
    old_fitted_positions = fitted_positions.copy() # before sigma clipping
    trace_residuals = np.array(trace_centre)-poly(row_array)

    # Clip 5 sigma outliers and refit
    std_residuals = mad(trace_residuals)
    clipped_trace_idx = (np.fabs(trace_residuals) <= 5*std_residuals)


    if len(row_array[clipped_trace_idx]) > 2:
        fitted_function = np.poly1d(np.polyfit(row_array[clipped_trace_idx],np.array(trace_centre)[clipped_trace_idx],trace_poly_order))
    else:
        fitted_function = poly

    y = np.arange(nrows)

    fitted_positions = fitted_function(np.arange(nrows))

    return fitted_positions, np.median(fwhm), np.array(gauss_std)


def explore_aperture(frame, trace):

    def plot_func(aperture_width, background_offset):

        plt.figure(figsize=(10, 5))
        vmin,vmax = np.nanpercentile(frame[0],[10,80])
        
        # Plot 2D transmission spectrum
        plt.imshow(frame, aspect='auto', cmap='viridis', vmin = vmin, vmax = vmax, origin = 'lower')
        
        # Plot trace center
        nrows = frame.shape[0]
        row_array = np.arange(nrows)
        plt.plot(trace, row_array, color='k', ls = '--', label = 'Trace Center')
        
        # Plot apertures
        plt.plot(trace - aperture_width*10, row_array, color='red', label = 'Aperture')
        plt.plot(trace + aperture_width*10, row_array, color='red')

        plt.legend()
        plt.xlabel(r'Row $\times$ 10')
        plt.ylabel('Column')

        plt.show()

    # Make the function interactive with a slider for aperture width
   
    interactive_plot = interactive(plot_func, 
                                aperture_width=widgets.IntSlider(min=1, max=20, step=1, value=10, description='Aperture Width'), 
                                background_offset=widgets.IntSlider(min=1, max=10, step=1, value=5, description='Background Offset'))

    display(interactive_plot)
    
    return 


def select_aperture_bg():
    """
    Function to select aperture width a background offset
    """
    aperture_width = int(input("Enter your preferred aperture width: "))
    background_offset = int(input("Enter your preferred background offset: "))
    return aperture_width, background_offset


def correct_1overf(data_model, aperture_width, background_offset, trace, row_min = 600, row_max = 2040, oversampling_factor=10):
    """
    Perform the correction for 1/f noise (group level background subtraction)

    Args:
        data_model : Post jump-step image with shape (ints, ngroups, nrows, ncols) in JWST datamodel format
        aperture_width (float): Width of extraction aperture in pixels
        background_offset (float): Offset from extraction aperture in pixels
        trace (list:float): List of floats of x positions defining the enter of the trace (note: cannot be array or jwst will not accept it)
        row_min (float): Minimum row of interest (usually 600 for nrs1)
        row_max (float): Maximum row of interest (usually 2040 for nrs1)
        oversampling_factor (float): Factor by which data was oversampled. Default is 10
    Returns:
        corrected image in original format with shape (ints, ngroups, nrows, ncols)
    """

   
    
    nints,ngroups,nrows,ncols = data_model.data.shape

     # Calculate the trace centre based on row min and row max
    trace = np.array(trace)
    trace_centre = nrows-np.round(trace/oversampling_factor).astype(int)
    col_min = row_min
    col_max = row_max

    # Make empty background mask to be filled
    bkg_mask = np.ones((nrows,ncols))
    
    # Loop through columns in background
    for i,col in enumerate(range(col_min,col_max)):
        # set non-background rows to zero in our mask

        # pdb.set_trace()
        bkg_mask[:,col][trace_centre[i]: trace_centre[i] + aperture_width//2 + background_offset] = 0
        bkg_mask[:,col][trace_centre[i] - aperture_width//2 - background_offset: trace_centre[i]] = 0


    one_over_f_noise = []

    # Loop through each integration
    for i in tqdm(range(nints)):

        group_flux = data_model.data[i]

        # Then each group
        for g in range(ngroups):

            image = group_flux[g]

            bkg = []

            # for non-NIRCam, the 1/f noise is along the columns not the rows. Here we're dealing with column-wise 1/f
            for c in range(ncols):

                bkg_pixels = image[:,c][bkg_mask[:,c].astype(bool)]

                # Take the median along each column
                col_median = np.median(bkg_pixels) 

                bkg.append(col_median)
                

            if g == ngroups - 1:
                one_over_f_noise.append(np.array(bkg))

            # Subtract background from the image
            image = image - np.array(bkg)

            data_model.data[i][g] = image

    return data_model





class Correct1OverFStep(Step):

    """ 
    Correcting the 1/f noise in an image with group level background subtraction and putting it back in jwst datamodel format
    """

    spec = """
    row_min = integer(default=0) # The integer value of the minimum row that you want to include. Can be found by looking at a DS9 image. NOTE: THIS CODE ASSUMES THAT SPECTRA ARE DISPERSED ALONG THE VERTICAL AXIS.
    row_max = integer(default=2040) # The integer value of the maximum row that you want to include. Can be found by looking at a DS9 image. NOTE: THIS CODE ASSUMES THAT SPECTRA ARE DISPERSED ALONG THE VERTICAL AXIS.
    aperture_width = float(default=10)
    background_offset = float(default=4)
    trace = float_list(default=None)
    oversampling_factor = float(default=10)
    """

    def process(self, input):
        """
        Perform the 1/f correction

        Args:
            data: JWST data model after jump step

        Returns:
            JWST data model with 1/f noise corrected
        """

        with datamodels.open(input) as model:

            # Make a copy of the model
            corrected_model = model.copy()

        # Perform 1/f subtraction
        corrected_model = correct_1overf(corrected_model, self.aperture_width, self.background_offset, self.trace, self.row_min, self.row_max)

        corrected_model.meta.cal_step.glbs_corrected = 'COMPLETE'

        return corrected_model






def stage1_nrs1(path_to_data, filename, path_to_crds = 'crds_cache/jwst_pub/references/jwst/nirspec/', perform_1overf = True, row_min = 600, row_max = 2040, oversampling_factor = 10, search_width = 10, guess_locations = 16, gaussian_width = 5, trace_poly_order = 4, choose_aperture = True, aperture_width = None, background_offset = None ):

    """
    Run stage 1 of the JWST pipeline optionally with Tiberius 1/f corrections for NRS1 (with updated crds files)

    Args:
        path_to_data (str): path to the JWST data to be processed (add slash at the end) e.g. './data/nrs1/'
        filename (str): fits filename of the JWST data at the _uncal.fits stage e.g. jwxxxxxx_04102_00001-seg001_nrs1_uncal.fits
        path_to_crds (str): path to crds files e.g. crds_cache/jwst_pub/references/jwst/nirspec/
        perform_1overf (bool): perform 1/f correction or not (True or False)
        row_min (float): Minimum row of interest (usually 600 for nrs1)
        row_max (float): Maximum row of interest (usually 2040 for nrs1)
        oversampling_factor (float): Factor by which data was oversampled. Default is 10
        search_width (float): the search width (in pixels) around 'trace_guess_locations' where the code will try to locate the star. For crowded fields, set this to a narrower value. For poor pointing (lost of drift) set this to a wider value. 
        guess_locations (float): approximate location of the centre of the star's trace in x-pixel coordinates (in the cross-dispersion axis, can be found via DS9). If more than one star is to be extracted, separate the guess locations with a comma ","
        gaussian_width (float): width of the Gaussian that is used to fit the trace location along each row in the cross-dispersion axis. This value doesn't matter too much and so can be kept at the default value of 5.
        trace_poly_order (float): Define the order of the polynomial used to fit the centroids of the Gaussians in the dispersion axis. i.e., what order polynomial best describes the curvature of the spectral trace? In most cases, the default of 4 is sufficient. If using a spline, this must be set to 0.
        choose_aperture (bool): Set to true if you want help selecting an aperture width and background offset with an interactive plot
        aperture_width (int): Aperture width in pixels. Leave blank if choose_aperture = True
        background_offset (int): Offset of background from aperture in pixels.  Leave blank if choose_aperture = True

    """

    # Instantiate all steps 
    stsci_group_scale = calwebb_detector1.group_scale_step.GroupScaleStep()
    stsci_dq_init = calwebb_detector1.dq_init_step.DQInitStep()
    stsci_saturation = calwebb_detector1.saturation_step.SaturationStep()
    stsci_superbias = calwebb_detector1.superbias_step.SuperBiasStep()
    stsci_refpix = calwebb_detector1.refpix_step.RefPixStep()
    stsci_linearity = calwebb_detector1.linearity_step.LinearityStep()
    stsci_dark_current = calwebb_detector1.dark_current_step.DarkCurrentStep()
    stsci_jump = calwebb_detector1.jump_step.JumpStep()
    stsci_ramp_fit = calwebb_detector1.ramp_fit_step.RampFitStep()
    stsci_gain_scale = calwebb_detector1.gain_scale_step.GainScaleStep()

    # Load the data 
    raw_data = datamodels.RampModel(os.path.join(path_to_data, filename))

    # Start the reduction up to jump step

    print('Running Group Scale')
    proc = stsci_group_scale.call(raw_data)
    print('Running DQ Init')
    proc = stsci_dq_init.call(proc, override_mask = f'{path_to_crds}jwst_nirspec_mask_0048.fits')
    print('Running Saturation')
    proc = stsci_saturation.call(proc, override_saturation = f'{path_to_crds}jwst_nirspec_saturation_0028.fits')
    print('Running Superbias')
    proc = stsci_superbias.call(proc, override_superbias = f'{path_to_crds}jwst_nirspec_superbias_0427.fits')
    print('Running Refpix')
    proc = stsci_refpix.call(proc, odd_even_columns=True)
    print('Running Linearity')
    proc = stsci_linearity.call(proc, override_linearity = f'{path_to_crds}jwst_nirspec_linearity_0024.fits')
    print('Running Dark Current')
    proc = stsci_dark_current.call(proc)
    print('Running Jump')
    proc = stsci_jump.call(
        proc, rejection_threshold=10.,
        flag_4_neighbors=True, min_jump_to_flag_neighbors=10.,
            three_group_rejection_threshold=15., four_group_rejection_threshold=15.,
            expand_large_events=False,
        skip=False)

    # If performing 1/f correction"
    if perform_1overf:

        print('Running 1/f Correction')
        # Get trace x positions
        frame = get_frame(raw_data, oversampling_factor = oversampling_factor, row_min = row_min, row_max = row_max)
        trace, fwhm, gauss_std = find_spectral_trace(frame, guess_locations, search_width, gaussian_width, trace_poly_order)

        # Get aperture width and background offset
        if choose_aperture:
            explore_aperture(frame[0], trace) # Plot
            aperture_width, background_offset = select_aperture_bg() # User inputs aperture

        # Perform 1/f correction
        proc = Correct1OverFStep.call(proc, row_min = row_min, row_max = row_max, aperture_width = aperture_width, background_offset = background_offset, trace = trace.tolist())

    print('Running Ramp Fit')
    # Continue from ramp fitting step 
    _, proc = stsci_ramp_fit.call(proc)
    print('Running Gain Scale')
    stage_1_output = stsci_gain_scale.call(proc, override_gain = f'{path_to_crds}jwst_nirspec_gain_0025.fits')
    
    # Save the output
    dir_path = path_to_data
    path = f"{filename.split('_uncal.fits')[-2]}_rateints.fits"

    # Save fits file
    stage_1_output.save(path = path, dir_path = dir_path)
    print(f'Saved as {os.path.join(dir_path, path)}')

    return proc
    

    



