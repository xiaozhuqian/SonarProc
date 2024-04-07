import sys
sys.path.append('.')
from utils import vis
from utils.vars import SHARED_VARS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image as Im
from datetime import datetime

import QC.quantity_results as q_result



class QualityControl(object):
    def __init__(self,bs_name,para,bs,geo_coords, angle, recorded_speed):
        self.name = bs_name.split('.')[-2]
        self.para = para
        self.raw_bs = bs
        self.geo_coords = geo_coords
        self.angle_name = self.para['angle']
        self.angle = angle
        self.recorded_speed = recorded_speed

        self.raw_bs_vis = vis.linearStretch(self.raw_bs, 1, 255, 0.02)
        self.ping_index = np.random.choice(bs.shape[0], 2, replace=False)
        
        self.raw_ping1 = self.raw_bs[self.ping_index[0]]
        self.raw_ping2 = self.raw_bs[self.ping_index[1]]

    def get_preprocessing_out(self, smoothed_proj_coords,
                              smoothed_angle,
                              smoothed_recorded_speed):
        self.smoothed_proj_coords = smoothed_proj_coords
        self.smoothed_angle = smoothed_angle
        self.smoothed_recorded_speed = smoothed_recorded_speed
        return

    def get_tvg_out(self, bs_tvg):
        self.bs_tvg = bs_tvg
        self.bs_tvg_vis = vis.linearStretch(self.bs_tvg, 1, 255, 0.02)
        self.tvg_ping1 = self.bs_tvg[self.ping_index[0]]
        self.tvg_ping2 = self.bs_tvg[self.ping_index[1]]
        return

    def get_bottom_detection_out(self, bottom_coords):
        self.bottom_coords = bottom_coords
        port_coord = bottom_coords[:, [0,2]].tolist()
        starboard_coord = bottom_coords[:, [1,2]].tolist()
        points = port_coord + starboard_coord
        color_img = vis.draw_points_on_image(points, self.bs_tvg_vis)
        self.bs_with_bottom_vis = color_img
        return
    
    def get_slant_corrected_out(self, bs_slant_corrected):
        self.bs_slant_corrected = bs_slant_corrected
        self.bs_slant_corrected_vis = vis.linearStretch(self.bs_slant_corrected, 1, 255, 0.02)
        self.flat_ping1 = self.bs_slant_corrected[self.ping_index[0]]
        self.flat_ping2 = self.bs_slant_corrected[self.ping_index[1]]
        return
    
    def get_gray_enhanced_out(self, bs_gray_enhanced):
        self.bs_gray_enhanced = bs_gray_enhanced
        self.bs_gray_enhanced_vis = vis.linearStretch(self.bs_gray_enhanced, 1, 255, 0.02)
        self.gray_enhanced_ping1 = self.bs_gray_enhanced[self.ping_index[0]]
        self.gray_enhanced_ping2 = self.bs_gray_enhanced[self.ping_index[1]]
        return

    def get_speed_corrected_out(self, bs_speed_corrected):
        self.bs_speed_corrected_vis = bs_speed_corrected
        return
    
    def get_geocoding_out(self, bs_geocoding, interping_proj_coords, interping_angle):
        self.bs_geocoding_vis = bs_geocoding
        self.interping_proj_coords = interping_proj_coords
        self.interping_angle = interping_angle
        return

    def get_time_cost(self, time_cost):
        self.time_cost = time_cost
        return

    def get_bottom_depth(self):
        sample_count = self.raw_bs.shape[1]
        port_bottom_depth, starboard_bottom_depth, flat_range = \
            q_result.cal_depth_flat_range(sample_count, 
                                          self.bottom_coords, 
                                                self.para['slant_range'])
        self.port_bottom_depth = port_bottom_depth
        self.starboard_bottom_depth = starboard_bottom_depth
        self.flat_swath = flat_range
        return
    
    def get_survey_line_length(self):
        smoothed_x = self.interping_proj_coords[:,0]
        smoothed_y = self.interping_proj_coords[:,1]
        survey_line_length = q_result.cal_survey_line_length(smoothed_x,smoothed_y)
        self.survey_line_length = survey_line_length
        return

    def get_scanning_area(self):
        area = q_result.cal_area(self.bs_geocoding_vis, self.para['geocoding_img_res'])
        self.area = area
        return

    def bs_plot(self):
        fig = plt.figure(figsize=(SHARED_VARS.PAGESIZE))

        ax1 = fig.add_subplot(711)
        ax1.imshow(np.rot90(self.raw_bs_vis,1),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        ax1.set_title('Raw waterfall')

        ax2 = fig.add_subplot(712)
        ax2.imshow(np.rot90(self.bs_tvg_vis,1),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        ax2.set_title('TVG enhanced waterfall')

        ax3 = fig.add_subplot(713)
        ax3.imshow(np.rot90(self.bs_with_bottom_vis[...,::-1],1))
        plt.xticks([])
        plt.yticks([])
        ax3.set_title('Waterfall with bottom line')

        ax4 = fig.add_subplot(714)
        ax4.imshow(np.rot90(self.bs_slant_corrected_vis,1),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        ax4.set_title('Slant corrected waterfall')

        ax5 = fig.add_subplot(715)
        ax5.imshow(np.rot90(self.bs_gray_enhanced_vis,1),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        ax5.set_title('Gray enhanced waterfall')

        ax6 = fig.add_subplot(716)
        ax6.imshow(np.rot90(self.bs_speed_corrected_vis,1),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        ax6.set_title('Speed corrected waterfall')

        ax7 = fig.add_subplot(717)
        ax7.imshow(np.rot90(self.bs_geocoding_vis,1),cmap='gray')
        plt.xticks([])
        plt.yticks([])
        ax7.set_title('Geocoded sidescan image')

        plt.suptitle("Waterfall after each processing step", fontsize=SHARED_VARS.TITLE_FONTSIZE)
        fig.tight_layout()
        # plt.savefig('./data/outputs2/bs.pdf',dpi=SHARED_VARS.PDF_DPI)
        # plt.show()

        return fig
    
    
    
    def ping_plots(self): 
        fig = plt.figure(figsize=(SHARED_VARS.PAGESIZE))
        ax1 = fig.add_subplot(521)
        ax1.plot(np.arange(len(self.raw_ping1)),self.raw_ping1)
        ax1.set_ylabel('Strength')
        ax1.set_title(f'Raw ping {self.ping_index[0]}')

        ax2 = fig.add_subplot(522)
        ax2.plot(np.arange(len(self.raw_ping2)),self.raw_ping2)
        ax2.set_title(f'Raw ping {self.ping_index[1]}')

        ax3 = fig.add_subplot(523)
        ax3.plot(np.arange(len(self.tvg_ping1)),self.tvg_ping1)
        ax3.set_ylabel('Strength')
        ax3.set_title(f'TVG enhanced ping {self.ping_index[0]}')

        ax4 = fig.add_subplot(524)
        ax4.plot(np.arange(len(self.tvg_ping2)),self.tvg_ping2)
        ax4.set_title(f'TVG enhanced ping {self.ping_index[1]}')

        ax5 = fig.add_subplot(525)
        port_bottom_coords = self.bottom_coords[self.ping_index[0],0]
        starboard_bottom_coords = self.bottom_coords[self.ping_index[0],1]
        ax5.plot(np.arange(len(self.tvg_ping1)),self.tvg_ping1)
        ax5.axvline(port_bottom_coords, color='red')
        ax5.axvline(starboard_bottom_coords, color='red')
        ax5.set_ylabel('Strength')
        ax5.set_title(f'Bottom line on TVG enhanced ping {self.ping_index[0]}')

        ax6 = fig.add_subplot(526)
        port_bottom_coords = self.bottom_coords[self.ping_index[1],0]
        starboard_bottom_coords = self.bottom_coords[self.ping_index[1],1]
        ax6.plot(np.arange(len(self.tvg_ping1)),self.tvg_ping2)
        ax6.axvline(port_bottom_coords, color='red')
        ax6.axvline(starboard_bottom_coords, color='red')
        ax6.set_title(f'Bottom line on TVG enhanced ping {self.ping_index[1]}')

        ax7 = fig.add_subplot(527)
        ax7.plot(np.arange(len(self.flat_ping1)),self.flat_ping1)
        ax7.set_ylabel('Strength')
        ax7.set_title(f'Slant corrected ping {self.ping_index[0]}')

        ax8 = fig.add_subplot(528)
        ax8.plot(np.arange(len(self.flat_ping2)),self.flat_ping2)
        ax8.set_title(f'Slant corrected ping {self.ping_index[1]}')

        ax9 = fig.add_subplot(529)
        ax9.plot(np.arange(len(self.gray_enhanced_ping1)),self.gray_enhanced_ping1)
        ax9.set_xlabel('Backscatter samples')
        ax9.set_ylabel('Strength')
        ax9.set_title(f'Gray enhanced ping {self.ping_index[0]}')

        ax10 = fig.add_subplot(5,2,10)
        ax10.plot(np.arange(len(self.gray_enhanced_ping2)),self.gray_enhanced_ping2)
        ax10.set_xlabel('Backscatter samples')
        ax10.set_title(f'Gray enhanced ping {self.ping_index[1]}')

        plt.suptitle("Backscatters of two random pings in each processing step", fontsize=SHARED_VARS.TITLE_FONTSIZE)
        fig.tight_layout()
        # plt.savefig('./data/outputs2/ping.pdf',dpi=SHARED_VARS.PDF_DPI)
        # plt.show()
        return fig
    
    

    def bottom_depth_plot(self):
        ping_count = self.raw_bs.shape[0]

        fig = plt.figure(figsize=(SHARED_VARS.PAGESIZE))

        ax1 = fig.add_subplot(121)
        ax1.plot(np.arange(ping_count),self.port_bottom_depth,'g-',label='Port')
        ax1.plot(np.arange(ping_count),self.starboard_bottom_depth,'r-',label='Starboard')
        ax1.set_ylabel('Depth (m)')
        ax1.set_xlabel('Ping index')
        ax1.legend(loc='upper right',ncol=1)
        #ax1.set_title('Bottom depth vs ping index')
        
        ax2 = fig.add_subplot(122)
        f = ax2.boxplot([self.port_bottom_depth,self.starboard_bottom_depth],
                    vert=True,
                    patch_artist=True)
        colors = ['green', 'red']
        for patch, color in zip(f['boxes'], colors):
            patch.set_facecolor(color)
        ax2.yaxis.grid(True) #在y轴上添加网格线
        ax2.set_ylabel('Depth (m)') #设置y轴名称
        ax2.set_xticks([1,2])
        ax2.set_xticklabels(['Port','Starboard'])

        fig.suptitle("Bottom depth distribution", fontsize=SHARED_VARS.TITLE_FONTSIZE)
        fig.tight_layout()
        # plt.savefig('./data/outputs2/depth.pdf',dpi=SHARED_VARS.PDF_DPI)
        # plt.show()

        return fig


    def image_histogram(self):

        fig = plt.figure(figsize=(SHARED_VARS.PAGESIZE))

        ax1 = fig.add_subplot(131)
        n, bins, patches = ax1.hist(self.raw_bs_vis.ravel(), 256, density=True) # the histogram of the data
        mu = np.mean(self.raw_bs_vis)
        sigma = np.std(self.raw_bs_vis)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
            np.exp(-0.5 * (1 / sigma * (bins[:-1] - mu))**2)) # add a 'best fit' line
        dif = q_result.hist_diff(n, y) #absolute difference between histgram and normal distribution
        ax1.plot(bins[:-1], y, '--')
        ax1.set_xlabel('Gray level')
        ax1.set_ylabel('Probability density')
        ax1.set_title('Raw waterfall: \n'
                    f'mu={mu:.0f}, sigma={sigma:.0f}, \ndifference={dif:.04f}')

        ax2 = fig.add_subplot(132)
        n, bins, patches = ax2.hist(self.bs_tvg_vis.ravel(), 256, density=True) # the histogram of the data
        mu = np.mean(self.bs_tvg_vis)
        sigma = np.std(self.bs_tvg_vis)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
            np.exp(-0.5 * (1 / sigma * (bins[:-1] - mu))**2)) # add a 'best fit' line
        dif = q_result.hist_diff(n, y) #absolute difference between histgram and normal distribution
        ax2.plot(bins[:-1], y, '--')
        ax2.set_xlabel('Gray level')
        ax2.set_ylabel('Probability density')
        ax2.set_title('TVG enhanced waterfall: \n'
                    f'mu={mu:.0f}, sigma={sigma:.0f}, \ndifference={dif:.04f}')
        
        ax3 = fig.add_subplot(133)
        n, bins, patches = ax3.hist(self.bs_gray_enhanced_vis.ravel(), 256, density=True)
        mu = np.mean(self.bs_gray_enhanced_vis)
        sigma = np.std(self.bs_gray_enhanced_vis)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
            np.exp(-0.5 * (1 / sigma * (bins[:-1] +20 - mu))**2))
        dif = q_result.hist_diff(n, y)
        ax3.plot(bins[:-1], y, '--')
        ax3.set_xlabel('Gray level')
        ax3.set_ylabel('Probability density')
        ax3.set_title('Gray enhanced waterfall: \n'
                    f'mu={mu:.0f}, sigma={sigma:.0f}, \ndifference={dif:.04f}')
        
        fig.suptitle('Histogram before and after gray enhancement',fontsize=SHARED_VARS.TITLE_FONTSIZE)
        
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.1) # add 10%h plot at bottom 

        fig.add_axes([0.025, 0.01, 0.95, 0.05]) # add axex, move right, up, width, height corresponding to above plot
        note_string = '- Note: difference: absolute difference between histgram and normal distribution'
        plt.text(0, 0, note_string, ha='left', va='bottom', wrap=True, fontsize=SHARED_VARS.TEXT_FONTSIZE)
        plt.axis('off')

        # plt.savefig('./data/outputs2/histgram.pdf',dpi=SHARED_VARS.PDF_DPI)
        # plt.show()
        
        return fig
    
    def geocoords_angle_plot(self):
        '''
        draw lon/lat, smoothed lon/lat; x/y, smoothed x/y,
            heading/smoothed heading (if exist), course/smoothed course (if exist), cog (if exist)
            recorded speed (if exist)
            
        '''
        raw_lon = self.geo_coords[:,0]
        raw_lat = self.geo_coords[:,1]
        smoothed_x = self.smoothed_proj_coords[:,0]
        smoothed_y = self.smoothed_proj_coords[:,1]
        interping_x = self.interping_proj_coords[:,0]
        interping_y = self.interping_proj_coords[:,1]
        smoothed_angles = self.smoothed_angle
        interping_angle = self.interping_angle

        fig = plt.figure(figsize=(SHARED_VARS.PAGESIZE))
        spec=fig.add_gridspec(nrows=2,ncols=2)
        ax1 = fig.add_subplot(spec[0,0])
        ax1.scatter(raw_lon,raw_lat,s=1,color='blue')
        ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
        x_tick = ax1.get_xticklabels()
        plt.setp(x_tick,rotation=30)
        ax1.set_xlabel('Lontitude (degree)')
        ax1.set_ylabel('Latitude (degree)')
        ax1.set_title('Raw survey line in geographic coordinates system')

        ax2 = fig.add_subplot(spec[0,1])
        ax2.scatter(smoothed_x,smoothed_y, s=1, label='Smoothed',color='blue')
        ax2.plot(interping_x, interping_y, linewidth=1, label = 'Interpolated',  color='purple')
        ax2.legend(loc='upper right',ncol=1)
        ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
        x_tick = ax2.get_xticklabels()
        plt.setp(x_tick,rotation=30)
        ax2.set_xlabel('Projected x (m)')
        ax2.set_ylabel('Projected y (m)')
        ax2.set_title('Survey line in projected coordinates system')

        ax3 = fig.add_subplot(spec[1,0])
        if self.angle_name == 'cog':
            ax3.scatter(np.arange(0,len(interping_angle)), interping_angle, s=1)
            angle_str = '- Cog (course over ground) is calculated from smoothed fish projected coordinates.'
        if self.angle_name == 'heading':
            ax3.scatter(np.arange(0, len(self.angle)), self.angle, s=1, label='Raw',color='blue')
            ax3.plot(np.arange(0, len(smoothed_angles)), smoothed_angles, linewidth=1, label = 'Smoothed',  color='purple')
            #ax3.scatter(np.arange(0, len(interping_angle)), interping_angle, s=1, label = 'interping')
            ax3.legend(loc='upper right',ncol=1)
            angle_smooth_factor = self.para['angle_smooth_factor']
            angle_str = f'- Heading is obtained from compass embedded in tow fish and is smoothed using a cubic spline curve with a smoothing factor: {angle_smooth_factor:.1f}.'
        if self.angle_name == 'course':
            ax3.scatter(np.arange(0, len(self.angle)), self.angle, s=1, label='Raw',color='blue')
            ax3.plot(np.arange(0, len(smoothed_angles)), smoothed_angles, linewidth=1, label = 'Smoothed',  color='purple')
            #ax3.scatter(np.arange(0, len(interping_angle)), interping_angle, s=1,label = 'interping')
            ax3.legend(loc='upper right',ncol=1)
            angle_smooth_factor = self.para['angle_smooth_factor']
            angle_str = f'- Course is obtained from satellite navigation sensor and is smoothed using a cubic spline curve with a smoothing factor: {angle_smooth_factor:.1f}.'
        ax3.set_xlabel('Ping index')
        y_label = self.angle_name.capitalize()
        ax3.set_ylabel(f'{y_label} (degree)')
        ax3.set_title('Ping direction along survey line')

        if self.recorded_speed is not None:
            ax4 = fig.add_subplot(spec[1,1])
            ax4.scatter(np.arange(0,len(self.recorded_speed)), self.recorded_speed, s=1, label='Raw',color='blue')
            ax4.plot(np.arange(0,len(self.smoothed_proj_coords)), self.smoothed_recorded_speed, linewidth=1, label='Smoothed', color='purple')
            ax4.legend(loc='upper right',ncol=1)
            ax4.set_xlabel('Ping index')
            ax4.set_ylabel('Vessel speed (knot)')
            ax4.set_title('Vessel speed along survey line')
            angle_smooth_factor = self.para['angle_smooth_factor']
            speed_str = f'- Vessel speed is obtained from satellite navigation sensor and is smoothed using a cubic spline curve with a smoothing factor: {angle_smooth_factor:.1f}.'
        else:
            speed_str = None

        fig.suptitle('Survey line and ping direction',fontsize=SHARED_VARS.TITLE_FONTSIZE)
        fig.tight_layout()

        plt.subplots_adjust(bottom=0.15) # add 10%h plot at bottom 

        fig.add_axes([0.025, 0.01, 0.95, 0.12]) # add axex, move right, up, width, height corresponding to above plot
        geocoords_smooth_factor = self.para['fish_trajectory_smooth_factor']
        geocoords_string = f'- The survey line is smoothed using a cubic spline curve with a smoothing factor of {geocoords_smooth_factor:.1f}. The length of survey line is {self.survey_line_length:.1f} m after smoothing.'
        plt.text(0, 0, geocoords_string+'\n'+angle_str+'\n'+speed_str, ha='left', va='bottom', wrap=True, fontsize=SHARED_VARS.TEXT_FONTSIZE)
        plt.axis('off')

        # plt.savefig('./data/outputs2/coord_angle_speed.pdf',dpi=SHARED_VARS.PDF_DPI)
        # plt.show()

        return fig
        #return plt.gcf()
    
    def summary(self):
        r_speed_u, r_speed_sigma, r_speed_max, r_speed_min = \
            q_result.cal_statistic(self.recorded_speed)
        raw_angle_u, raw_angle_sigma, raw_angle_max, raw_angle_min = \
            q_result.cal_statistic(self.angle)
        smoothed_angle_u, smoothed_angle_sigma, smoothed_angle_max, smoothed_angle_min = \
            q_result.cal_statistic(self.smoothed_angle)
        port_altitude_u, port_altitude_std, port_altitude_max, port_altitude_min = \
            q_result.cal_statistic(self.port_bottom_depth)
        starboard_altitude_u, starboard_altitude_std, starboard_altitude_max, starboard_altitude_min = \
            q_result.cal_statistic(self.port_bottom_depth)

        fish_lon = self.geo_coords[:,0]
        fish_lat = self.geo_coords[:,1]

        date_now = datetime.now()
        date_str = \
        str(f'Run date: {date_now}\n'
            f'Time cost: {self.time_cost:.2f} s\n'

        )

        basic_str = \
        str('Environment condition:\n'
            f'- temperature (°C): {self.para["temperature"]}\n'
            f'- salinity (ppt): {self.para["salinity"]}\n'
            f'- depth (m): {self.para["depth"]}\n'
            f'- pH: {self.para["pH"]}\n'
            'Raw waterfall:\n'
            f'- frequency (kHz): {self.para["frequency"]}\n'
            f'- slant range (m): {self.para["slant_range"]}\n'
            f'- sample count: {self.raw_bs.shape[1]}\n'
            f'- ping count: {self.raw_bs.shape[0]}\n'
            'NMEA:\n'
            f'- planned speed (knot): {self.para["planed_speed"]}\n'
            f'- recorded speed (knot): mean={r_speed_u}, std={r_speed_sigma}, min={r_speed_min}, max={r_speed_max}\n'
            f'- raw tow fish geographic coordinates: \n  - min-max lontitude={fish_lon.min():.5f}-{fish_lon.max():.5f} \n  - min-max latitude={fish_lat.min():.5f}-{fish_lat.max():.5f}\n'
            f'- ping direction type: {self.angle_name}\n'
            f'- raw ping direction (degree): mean={raw_angle_u}, std={raw_angle_sigma}, min={raw_angle_min}, max={raw_angle_max}\n'
            f'- smoothed ping direction(degree): mean={smoothed_angle_u}, std={smoothed_angle_sigma}, min={smoothed_angle_min}, max={smoothed_angle_max}\n'
            'Tow fish:\n'
            f'- port altitude (m): mean={port_altitude_u:.1f}, std={port_altitude_std:.1f}, min={port_altitude_min:.1f}, max={port_altitude_max:.1f}\n'
            f'- starboard altitude (m): mean={starboard_altitude_u:.1f}, std={starboard_altitude_std:.1f}, min={starboard_altitude_min:.1f}, max={starboard_altitude_max:.1f}\n'
            'Geocoding mosaic:\n'
            f'- resolution (m/pixel): {self.para["geocoding_img_res"]}\n'
            f'- geographic EPSG: {self.para["geo_EPSG"]}, projected EPSG: {self.para["proj_EPSG"]}\n'
            f'- survey line length (smoothed, m): {self.survey_line_length:.1f}\n'
            f'- flat swath (port+starboard, m): min-max={self.flat_swath.min():.1f}-{self.flat_swath.max():.1f}\n'
            f'- scanning area (square meter): {self.area:.1f}\n'
            )

        para_str = \
        str('TVG enhancement:\n'
            f'- lambd: {self.para["lambd"]}\n'
            'Bottom detection:\n'
            f'- gradient_thred_factor: {self.para["gradient_thred_factor"]}\n'
            f'- win_h: {self.para["win_h"]}\n'
            f'- use_ab_line_correction: {self.para["use_ab_line_correction"]}\n'
            f'- peak_factor: {self.para["peak_factor"]}\n'
            f'- valley_std: {self.para["valley_std"]}\n'
            'Slant correction:\n'
            f'- bottom_offset: {self.para["bottom_offset"]}\n'
            'Gray enhancement:\n'
            f'- gray_enhance_method: {self.para["gray_enhance_method"]}\n'
            f'- gaussian_kernel: {self.para["gaussian_kernel"]}\n'
            f'- gain: {self.para["gain"]}\n'
            f'- alpha: {self.para["alpha"]}\n'
            f'- ratio: {self.para["ratio"]}\n'
            f'- prob_thred: {self.para["prob_thred"]}\n'
            'Geocoding:\n'
            f'- is_ns: {self.para["is_ns"]}\n'
            f'- fish_trajectory_smooth_type: {self.para["fish_trajectory_smooth_type"]}\n'
            f'- fish_trajectory_smooth_factor: {self.para["fish_trajectory_smooth_factor"]}\n'
            f'- angle: {self.para["angle"]}\n'
            f'- angle_smooth_factor: {self.para["angle_smooth_factor"]}\n')


        fig = plt.figure(figsize=SHARED_VARS.PAGESIZE)
        # spec=fig.add_gridspec(nrows=1,ncols=2,width_ratios=[2,1])

        plt.axis([0,1,0,1])
        fig.tight_layout()

        plt.text(0, 0.95, date_str, ha='left', va='top', wrap=True, fontsize=SHARED_VARS.TEXT_FONTSIZE)
        plt.text(0, 0.88, 'BASIC INFORMATION', ha='left', va='top', wrap=True, fontsize=SHARED_VARS.SUBTITLE_FONTSIZE)
        plt.text(0, 0.85, basic_str, ha='left', va='top', wrap=True, fontsize=SHARED_VARS.TEXT_FONTSIZE)
        plt.text(0.65, 0.88, 'PROCESSING PARAMETERS', ha='left', va='top', wrap=True, fontsize=SHARED_VARS.SUBTITLE_FONTSIZE)
        plt.text(0.65, 0.85, para_str, ha='left', va='top', wrap=True, fontsize=SHARED_VARS.TEXT_FONTSIZE)
        plt.axis('off')

        fig.suptitle(f"Summary of {self.name}", fontsize=SHARED_VARS.TITLE_FONTSIZE)
        

        # plt.savefig('./data/outputs2/summary.pdf',dpi=SHARED_VARS.PDF_DPI)
        # plt.show()

        return fig

def gen_pdf(path, plot_dict):
    with PdfPages(path) as pdf:
        for name, plot in plot_dict.items():
            pdf.savefig(plot)
    return





        
        



        











