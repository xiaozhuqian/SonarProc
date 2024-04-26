if __name__ == '__main__':
    '''
    Preprocessing waterfall images, including bottom detection, slant correction, gray enhancement, speed correction, and mosaicing/geocoding.
    Inputs:
        downsampled backscatter data: *_frequency-bs.npy;
        survey line geographical coordinates: *_frequency-geo_coords.npy
        vessel speed (optional): *_frequency-nmeas.npy 
        vessel course (optional): *_frequency-nmeas.npy
        towfish heading (optional): *_frequency-attitudes.npy
    Outputs:
        Final outputs:
            corrected waterfall images: *_20-bs-speed_correction.png
            tile mosaics: *_20-bs-geocoding.png
            QC reports: *_20-bs-quality_control.pdf
            processing log: *.log
        intermediate outputs:
            bottom cooridinates, slant corrected images, gray enhanced images, etc.
    '''

    import argparse
    import os
    import glob
    import time
    import cv2
    import logging
    import numpy as np

    import utils.logger as logger
    from utils.vis import gray_enhancing_tvg, npy2img, draw_points_on_image, linearStretch
    from utils.util import absoption
    import prep.preprocessing as prepp

    from bottom_detection.bottom_detect import bottom_detection
    from bottom_detection.ab_line_detection import *
    from geo_correction.slant_correct import geo_correction
    from geo_correction.speed_correct import blockwise_speed_correction, overall_speed_correction
    from gray_enhancement.gray_enhance import gray_enhancing_retinex, gray_enhancing_statistic, coarse2fine
    from geocoding.coords_transform_based_cal import geocoding
    from geocoding.save_raster import save_geotiff
    from QC.qc_vis import QualityControl,gen_pdf
    

    parser = argparse.ArgumentParser(description='Waterfall preprocessing and mosaicing')
    parser.add_argument('--input_dir', default='./outputs/npy/', help='directory of input file')
    parser.add_argument('--out_dir', default='./outputs/tile', help='dierctory to save output')
    
    parser.add_argument('--frequency', default=20, type=int, help='frequency code, high frequency (1600 kHz)=21, low frequency (600 kHz) = 20')
    parser.add_argument('--slant_range', default=80., type=float, help='slant_range')

    parser.add_argument('--planed_speed', default=4., type=float, help='vessel speed (knots)')
    parser.add_argument('--recorded_speed', default=1, type=int, help='0: do not have recorded_speed, 1: have recorded_speed')

    # tvg parameters
    parser.add_argument('--temperature', default=26., type=float, help='sea water temperature')
    parser.add_argument('--salinity', default=28., type=float, help='sea water salinity')
    parser.add_argument('--depth', default=5., type=float, help='towfish depth')
    parser.add_argument('--pH', default=8., type=float, help='sea water pH')
    parser.add_argument('--lambd', default=20., type=float, help='tvg log coefficient')
    
    # bottom detection parameters
    parser.add_argument('--gradient_thred_factor', default=0.1, type=float, help='coefficient of maximum gradient')
    parser.add_argument('--win_h', default=10, type=int, help='abnormal detection window height/2')
    
    parser.add_argument('--use_ab_line_correction', default=0, type=int, help='abnormal line correction flag, if use, 1, esle 0')
    parser.add_argument('--peak_factor', default=0.7, type=float, help='coefficient of peak detection') # work when use_ab_line_correction=1
    parser.add_argument('--valley_std', default=5, type=int, help='valley detection standard deviation')# work when use_ab_line_correction=1

    # slant correction parameters
    parser.add_argument('--bottom_offset', default=[2,5], type=int, help='bottom coordinates offsets of port and starboard, positive is plus')

    # gray enhancement parameters
    parser.add_argument('--gray_enhance_method', default='coarse2fine', help='retinex, statistic, coarse2fine')
    parser.add_argument('--gaussian_kernel', default=50, type=int, help='gaussian filter kernel') # for retinex gray enhancement
    parser.add_argument('--gain', default=10., type=float, help='multiplication factor') # for retinex gray enhancement
    parser.add_argument('--alpha', default=0.05, type=float, help='intensity equalization factor') # for retinex gray enhancement
    parser.add_argument('--ratio', default=0.02, type=float, help='gray linear strech clip ratio') # for coarse2fine gray enhancement
    parser.add_argument('--prob_thred', default=0.3, type=float, help='no object pings detect probability threshold') # for coarse2fine gray enhancement, smaller, easier including objects

    # speed correction parameters
    parser.add_argument('--speed_correction_method', default='blockwise', help='blockwise, overall')

    # geocoding parameters
    parser.add_argument('--geo_EPSG', default=4490, type=int, help='geographic coordinate system')
    parser.add_argument('--proj_EPSG', default=4499, type=int, help='projection coordinate system')
    parser.add_argument('--geocoding_img_res', default=0.1, type=float, help='geocoded image resolution')
    parser.add_argument('--is_ns', default=1, type=int, help='survey line direction, for fill: 1 when north-west direction, 0 when east-west direction')
    parser.add_argument('--fish_trajectory_smooth_type', default='bspline', type=str, help='smooth method; bspline, linear')
    parser.add_argument('--fish_trajectory_smooth_factor', default=80., type=float, help='larger, smoother')
    parser.add_argument('--angle', default='cog', type=str, help='ping direction, [cog, heading, course]')
    parser.add_argument('--angle_smooth_factor', default=10., type=float, help='larger, smoother')
    
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir) 

    save_files = {'bottom_detection':'bottom_detection','slant_correction':'slant_correction',\
                  'gray_enhancement':'gray_enhancement','speed_correction':'speed_correction',\
                    'geocoding':'geocoding'}
    for file in save_files.values():
        if not os.path.exists(os.path.join(args.out_dir,file)):
            os.makedirs(os.path.join(args.out_dir,file)) 

    now = int(round(time.time()*1000))
    now02 = time.strftime('%Y-%m-%d-%H%M%S',time.localtime(now/1000))
    logger.setlogger(os.path.join(args.out_dir, f'{now02}-tile_construction.log'))  # set logger
    for k, v in args.__dict__.items():  # save args
        logging.info("{}: {}".format(k, v))
    

    bs_names = []
    for bs_path in sorted(glob.glob(os.path.join(args.input_dir, 'downsample', f'*{args.frequency}-bs.npy'))):
        _, bs_name = os.path.split(bs_path)
        bs_names.append(bs_name)

    for n, bs_name in enumerate(bs_names):
        tile_start = time.time()
        # read variables
        start = time.time()
        bs=np.load(os.path.join(args.input_dir, 'downsample', bs_name))
        bs_index = bs_name.split('.')[-2]
        geo_coords = np.load(os.path.join(args.input_dir, 'raw',bs_name).replace('bs','geo_coords')) #(ping_count, 2), (lon,lat)
        geo_coords = geo_coords[:,:2]

        if args.recorded_speed == 1:
            recorded_speed = np.load(os.path.join(args.input_dir, 'raw',f'{bs_index}.npy').replace('bs','nmeas'))  #(spped, course)
            recorded_speed = recorded_speed[:,0]
        else:
            recorded_speed = None
            
        if args.angle == 'heading':
            heading = np.load(os.path.join(args.input_dir, 'raw',f'{bs_index}.npy').replace('bs','attitudes')) #(heading, pitch, roll, ...)
            angle = heading[:,0]
        if args.angle == 'course':
            course = np.load(os.path.join(args.input_dir, 'raw',f'{bs_index}.npy').replace('bs','nmeas'))  #(spped, course)
            angle = course[:,1]
        if args.angle == 'cog':
            angle = None

        # bs = bs[:100,:]
        # geo_coords = geo_coords[:100,:]
        # angle = angle[:100]
        # recorded_speed = recorded_speed[:100]

        n_sample = bs.shape[1]
        n_pings = bs.shape[0]

        para = vars(args)
        qc = QualityControl(bs_name,para,bs,geo_coords,angle,recorded_speed)

        logging.info(f'{n+1}th image/total {len(bs_names)} images, {bs_name}, shape: {bs.shape}')
        
        # preprocessing
        proj_coords, coords_tck, coords_u = \
            prepp.coords_prep(geo_coords, 
                              args.fish_trajectory_smooth_type, 
                                args.fish_trajectory_smooth_factor,
                                    args.geo_EPSG, args.proj_EPSG) #clean, geographic coordinates to projected coordinates, smooth; (coords_tck, coords_u): smooth parameters
        
        if recorded_speed is not None:
            recorded_speed, _, _ = prepp.sequence_prep(recorded_speed,args.angle_smooth_factor)
        
        if angle is not None:
            angle, angle_tck, angle_u = prepp.sequence_prep(angle,args.angle_smooth_factor) #clean, smooth; (angle_tck, angle_u): smooth factor
        else:
            angle_tck = None 
            angle_u = None

        qc.get_preprocessing_out(proj_coords, angle, recorded_speed)

        # tvg enhancement
        if args.frequency == 20:
            frequency = 600
        if args.frequency == 21:
            frequency = 1600
        a = absoption(frequency*1000., args.temperature, args.salinity, args.depth, args.pH) # sound absoption in seawater, db/m
        bs_tvg = gray_enhancing_tvg(bs, args.slant_range, args.lambd, a)
        
        qc.get_tvg_out(bs_tvg)

        # bottom detection
        print('bottom detection ...')
        start = time.time()
        coarse_coords, ab_corrected_coords, smoothed_coords = \
            bottom_detection(bs_tvg, args.gradient_thred_factor, args.win_h)
          
        # abnormal line detection and correction
        ab_line_inter = False
        if args.use_ab_line_correction:
            #abnormal line detect
            coords_port = smoothed_coords[:, [0,2]]
            peaks = peak_detection(coords_port, n_sample, args.peak_factor)
            #abnormal line correction
            if peaks.shape[0]:
                coords_inter = ab_line_correction(smoothed_coords, n_sample, args.peak_factor, args.valley_std)
                ab_line_inter = True
                coord = coords_inter
            else:
                coord = smoothed_coords
        else:
            coord = smoothed_coords

        end = time.time()
        logging.info(f'{n+1}th image/total {len(bs_names)} images, {bs_name}, bottom_detection, cost: {end-start:.3f}')

        qc.get_bottom_detection_out(coord)

        # save bottom file
        img = linearStretch(bs_tvg, 1, 255, 0.02)
        
        if args.use_ab_line_correction and ab_line_inter:
            coords = [coarse_coords, ab_corrected_coords, smoothed_coords, coords_inter]
            name = ['1coarse', '2ab', '3smooth', '4inter']
        else:
            coords = [coarse_coords, ab_corrected_coords, smoothed_coords]
            name = ['1coarse', '2ab', '3smooth',]

        for i, coord in enumerate(coords):
            port_coord = coord[:, [0,2]].tolist()
            starboard_coord = coord[:, [1,2]].tolist()
            points = port_coord + starboard_coord
            color_img = draw_points_on_image(points, img)
            cv2.imwrite(os.path.join(args.out_dir,save_files['bottom_detection'],f'{bs_index}-bottom_{name[i]}.png'), color_img)
            np.savetxt(os.path.join(args.out_dir,save_files['bottom_detection'],f'{bs_index}-bottom_{name[i]}.txt'),coord, delimiter=' ',fmt='%d')
       
        # slant correction 
        print('geometric correction ...')
        start = time.time()
        bs_geo_corrected, (min_port_width, min_starboard_width) = geo_correction(bs_tvg, coord, args.bottom_offset) #slant correction
        bs_geo_corrected_vis = linearStretch(bs_geo_corrected, 1, 255, 0.02)
        
        end = time.time()
        logging.info(f'{n+1}th image/total {len(bs_names)} images, {bs_name}, slant_correction, cost: {end-start:.3f}')
        qc.get_slant_corrected_out(bs_geo_corrected)

        cv2.imwrite(os.path.join(args.out_dir, save_files['slant_correction'],f'{bs_index}-geocorrection.png'), bs_geo_corrected_vis)
        

        # gray enhancement
        print('gray enhancement ...')
        start = time.time()
        if args.gray_enhance_method == 'retinex':
            bs_gray_enhancement, bs_gray_enhancement_vis = gray_enhancing_retinex(bs_geo_corrected, args.gaussian_kernel, args.gain, args.alpha)
        if args.gray_enhance_method == 'statistic':
            bs_gray_enhancement, bs_gray_enhancement_vis = gray_enhancing_statistic(bs_geo_corrected, min_port_width, min_starboard_width, ratio=args.ratio)
        if args.gray_enhance_method == 'coarse2fine':
            bs_gray_enhancement, bs_gray_enhancement_vis, no_object_pings = coarse2fine(bs_geo_corrected, min_port_width, min_starboard_width, args.prob_thred,args.ratio)
        
        end = time.time()
        logging.info(f'{n+1}th image/total {len(bs_names)} images, {bs_name}, gray_enhancement, cost: {end-start:.3f}')

        qc.get_gray_enhanced_out(bs_gray_enhancement)
        #qc.image_histogram()

        cv2.imwrite(os.path.join(args.out_dir, save_files['gray_enhancement'],f'{bs_index}-grayenhance.png'), bs_gray_enhancement_vis)
        #cv2.imwrite(os.path.join(args.out_dir, save_files['gray_enhancement'],f'{bs_index}-noobjecpings.png'), no_object_pings)

        # speed correction
        print('speed correction ...')
        start = time.time()
        if args.speed_correction_method == 'blockwise':
            speed_corrected_bs_vis = blockwise_speed_correction(bs_gray_enhancement_vis, geo_coords, args.geocoding_img_res, args.geo_EPSG, args.proj_EPSG) #speed correction
        if args.speed_correction_method == 'overall':
            speed_corrected_bs_vis = overall_speed_correction(bs_gray_enhancement_vis, geo_coords, args.geocoding_img_res, args.geo_EPSG, args.proj_EPSG) #speed correction
        
        end = time.time()
        logging.info(f'{n+1}th image/total {len(bs_names)} images, {bs_name}, speed_correction, cost: {end-start:.3f}')

        qc.get_speed_corrected_out(speed_corrected_bs_vis)
        cv2.imwrite(os.path.join(args.out_dir, save_files['speed_correction'],f'{bs_index}-speed_correction.png'), speed_corrected_bs_vis)

        #geocoding
        print('geocoding ...')
        start = end = time.time()
        geocoding_img, upperleft_x, upperleft_y, proj_interping_coords, angle_interping = \
            geocoding(bs_gray_enhancement_vis, 
                      proj_coords, coords_tck, coords_u, 
                      n_sample, 
                      args.slant_range, args.geocoding_img_res, args.is_ns, 
                      args.fish_trajectory_smooth_type, args.angle, angle, angle_tck, angle_u)
        
        end = time.time()
        logging.info(f'{n+1}th image/total {len(bs_names)} images, {bs_name}, geocoding, cost: {end-start:.3f}')

        qc.get_geocoding_out(geocoding_img, proj_interping_coords, angle_interping)

        # save geo tiff
        save_geotiff(geocoding_img, upperleft_x, upperleft_y, 
                     args.geocoding_img_res, args.geo_EPSG, args.proj_EPSG,
                     os.path.join(args.out_dir, save_files['geocoding'],f'{bs_index}-geocoding.tif'))

        time_cost = time.time()-start
        qc.get_time_cost(time_cost)

        #get metrics
        qc.get_bottom_depth()
        qc.get_survey_line_length()
        qc.get_scanning_area()

        #plot
        bs_plot = qc.bs_plot()
        ping_plot = qc.ping_plots()
        depth_plot = qc.bottom_depth_plot()
        hist_plot = qc.image_histogram()
        coord_plot = qc.geocoords_angle_plot()
        summary_plot = qc.summary()

        qc_report = {'summary':summary_plot,'bs_plot':bs_plot,
                     'ping_plot':ping_plot, 'depth_plot':depth_plot,
                     'hist_plot':hist_plot,'coord_plot':coord_plot}
        gen_pdf(os.path.join(args.out_dir, f'{bs_index}-quality_control.pdf'),qc_report)

        tile_end = time.time()
        logging.info(f'{n+1}th image/total {len(bs_names)} images, {bs_name}, whole_tile, cost: {tile_end-tile_start:.3f}')
