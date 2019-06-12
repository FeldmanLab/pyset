#AC Coulomb Blockade Oscilations SET
# Copyright (C) 2019  Carlos Kometter

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
[info]
version = 1.0
'''
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import labrad
import labrad.units as U
import numpy as np
import math
import time
import yaml

X_MAX = 10.0
X_MIN = -10.0
Y_MAX = 10.0
Y_MIN = -10.0
ADC_AVGSIZE = 1
#ADC_CONVERSIONTIME = 250

adc_offset = np.array([0.0, 0.0])
adc_slope = np.array([1.0, 1.0])

def safety_check(meas_parameters, max_values, time_factor):
    # Ramp Vgate settings check
    vgate_rng = (meas_parameters['vgate_rng'][1] - meas_parameters['vgate_rng'][0])
    vgate_rate = vgate_rng / (meas_parameters['vgate_pnts'] * meas_parameters['delay'] * time_factor)
    if vgate_rate > max_values['vgate_rate']:
        raise ValueError('Vgate rate too high: ' + str(vgate_rate) + ' V/s.')

    vgate_step_size = vgate_rng / meas_parameters['vgate_pnts']
    if vgate_step_size > max_values['vgate_step_size']:
        raise ValueError('Vgate step size too high: ' + str(vgate_step_size) + ' V.')

    return 0


def time_factor(unit):
    if unit == 0:
        return 1e-6
    if unit == 1:
        return 1e-3


def create_file(dv, cfg, **kwargs): # try kwarging the vfixed
    try:
        dv.mkdir(cfg['file']['data_dir'])
        print("Folder {} was created".format(cfg['file']['data_dir']))
        dv.cd(cfg['file']['data_dir'])
    except Exception:
        dv.cd(cfg['file']['data_dir'])

    measurement = cfg['measurement']
    var_name1 = cfg[measurement]['v1']
    var_name2 = cfg[measurement]['v2']
    dac_adc = cfg['dacadc_settings']

    plot_parameters = {'extent': [cfg['meas_parameters']['vset_rng'][0],
                                  cfg['meas_parameters']['vset_rng'][1],
                                  cfg['meas_parameters']['vgate_rng'][0],
                                  cfg['meas_parameters']['vgate_rng'][1]],
                       'pxsize': [cfg['meas_parameters']['vset_pnts'],
                                  cfg['meas_parameters']['vgate_pnts']]
                      }

    dv.new(cfg['file']['file_name']+"-plot", ("i", "j", var_name1, var_name2),
           ('DC', 'AC', 'D', 'N', 'X', 'Y', 't'))
    print("Created {}".format(dv.get_name()))

    # Adding commens and parameters
    dv.add_comment(cfg['file']['comment'])

    measurement_parameters = cfg[measurement].keys()
    for parameter in measurement_parameters:
        dv.add_parameter(parameter, cfg[measurement][parameter])

    lockin_parameters = cfg['lockin_settings'].keys()
    for parameter in lockin_parameters:
        dv.add_parameter(parameter, cfg['lockin_settings'][parameter])

    dv.add_parameter('delay_unit', dac_adc['delay_unit'])

    dv.add_parameter('vset_rng', tuple(cfg['meas_parameters']['vset_rng']))
    dv.add_parameter('vset_pnts', cfg['meas_parameters']['vset_pnts'])
    dv.add_parameter('vgate_rng', tuple(cfg['meas_parameters']['vgate_rng']))
    dv.add_parameter('vgate_pnts', cfg['meas_parameters']['vgate_pnts'])
    dv.add_parameter('extent', tuple(plot_parameters['extent']))
    dv.add_parameter('pxsize', tuple(plot_parameters['pxsize']))
    dv.add_parameter('live_plots', (('vgate', 'vset', 'DC'), ('vgate', 'DC'), ('vgate', 'vset', 'AC'), ('vgate', 'AC')))
    #dv.add_parameter('plot', cfg['plot'])

    if kwargs is not None:
        for key, value in kwargs.items():
            dv.add_parameter(key, value)


def mesh(offset, xrange, yrange, pxsize=(100, 100)):
    """
    xrange and yrange are tuples (xmin, xmax) and (ymin, ymax)
    offset  is a tuple of offsets:  (X, Y)
    pxsize  is a tuple of # of steps:  (x steps, y steps)
    """
    y = np.linspace(yrange[0], yrange[1], pxsize[1]) - offset[1]
    x = np.linspace(xrange[0], xrange[1], pxsize[0]) - offset[0]
    x, y = np.meshgrid(x, y) 
    return np.dstack((x, y))

def main():

    # Loads config
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    measurement = cfg['measurement']
    measurement_settings = cfg[measurement]
    max_values = cfg['max_values']


    ## DacAdc settings

    dacadc_settings = cfg['dacadc_settings']
    timeout = dacadc_settings['timeout']
    dac1_ch = dacadc_settings['dac1_ch']
    dac2_ch = dacadc_settings['dac2_ch']
    adc1_ch = dacadc_settings['adc1_ch']
    adc2_ch = dacadc_settings['adc2_ch']
    delay_unit = dacadc_settings['delay_unit']
    t_factor = time_factor(delay_unit)

    # Meas parameters
    meas_parameters = cfg['meas_parameters']
    delay_meas = meas_parameters['delay']

    # Ramp settings to first point
    vgate2start_pnts = abs(int(meas_parameters['vgate_rng'][0] / max_values['vgate_step_size']))
    if vgate2start_pnts < 2: vgate2start_pnts = 3
    vgate2start_delay = abs(int(meas_parameters['vgate_rng'][0] / (vgate2start_pnts * max_values['vgate_rate']) * 1e6))
    if vgate2start_delay < 1: vgate2start_delay = 10

    vset2start_pnts = abs(int(meas_parameters['vset_rng'][0] / max_values['vset_step_size']))
    if vset2start_pnts < 2: vset2start_pnts = 3
    vset2start_delay = abs(int(meas_parameters['vset_rng'][0] / (vset2start_pnts * max_values['vset_rate']) * 1e6))
    if vset2start_delay < 1: vset2start_delay = 10

    # Ramp settings to next line
    # Vgate ramps back whole range
    vgate2next_pnts = abs(int((meas_parameters['vgate_rng'][1] - meas_parameters['vgate_rng'][0]) / max_values['vgate_step_size']))
    if vgate2next_pnts < 2: vgate2next_pnts = 3
    vgate2next_delay = abs(int((meas_parameters['vgate_rng'][1] - meas_parameters['vgate_rng'][0]) / (vgate2next_pnts * max_values['vgate_rate']) * 1e6))
    if vgate2next_delay < 1: vset2next_delay = 10

    # Vset ramps up to next line
    vset2next_pnts = abs(int((meas_parameters['vset_rng'][1] - meas_parameters['vset_rng'][0]) / meas_parameters['vset_pnts'] / max_values['vset_step_size']))
    if vset2next_pnts < 2: vset2next_pnts = 3
    vset2next_delay = abs(int((meas_parameters['vset_rng'][1] - meas_parameters['vset_rng'][0]) / meas_parameters['vset_pnts'] / (vset2next_pnts * max_values['vset_rate']) * 1e6))
    if vset2next_delay < 1: vset2next_delay = 10


    # Ramp settings to zero
    vgate2zero_pnts = abs(int(meas_parameters['vgate_rng'][1] / max_values['vgate_step_size']))
    if vgate2zero_pnts < 2: vgate2zero_pnts = 3
    vgate2zero_delay = abs(int(meas_parameters['vgate_rng'][1] / (vgate2zero_pnts * max_values['vgate_rate']) * 1e6))
    if vgate2zero_delay < 1: vgate2zero_delay = 10

    vset2zero_pnts = abs(int(meas_parameters['vset_rng'][1] / max_values['vset_step_size']))
    if vset2zero_pnts < 2: vset2zero_pnts = 3
    vset2zero_delay = abs(int(meas_parameters['vset_rng'][1] / (vset2zero_pnts * max_values['vset_rate']) * 1e6))
    if vset2zero_delay < 1: vset2zero_delay = 10

    # Safety check
    safety_check(meas_parameters, cfg['max_values'], t_factor)


    # Lockin settings
    lockin_settings = cfg['lockin_settings']
    tc_var = lockin_settings['tc']
    sens_var = lockin_settings['sensitivity']

    if delay_meas* t_factor < 3*tc_var:
        print("Warning: delay is less than 3x lockin time constant.")
    
    # Labrad connections and Instrument Configurations

    cxn = labrad.connect()
    reg = cxn.registry
    dv = cxn.data_vault
    dac_adc = cxn.dac_adc
    dac_adc.select_device()
    dac_adc.initialize()
    dac_adc.delay_unit(delay_unit)
    dac_adc.timeout(U.Value(timeout, 's'))
    dac_adc.read()
    dac_adc.read()

    # Create datavault data set
    create_file(dv, cfg)

    # Mesh parameters

    pxsize = (meas_parameters['vgate_pnts'], meas_parameters['vset_pnts'])
    extent = (meas_parameters['vgate_rng'][0], meas_parameters['vgate_rng'][1],
              meas_parameters['vset_rng'][0], meas_parameters['vset_rng'][1])
    num_x = pxsize[0]
    num_y = pxsize[1]
    print(extent, pxsize)

    # Start timer
    time.sleep(3)
    t0 = time.time()

    # Estimated time
    est_time = (num_x * num_y * delay_meas * t_factor + vgate2next_pnts * vgate2next_delay * 1e-6 * num_y) / 60.0
    dt = num_x*delay_meas*t_factor/60.0
    print("Will take a total of {} mins. With each line trace taking {} This deprecated for SET.".format(est_time, dt))

    m = mesh(offset=(0.0, -0.0), xrange=(extent[0], extent[1]),
             yrange=(extent[2], extent[3]), pxsize=pxsize)

    mdn = m # for future implementation

    for i in range(num_y):

        data_x = np.zeros(num_x)
        data_y = np.zeros(num_x)

        vec_x = m[i, :][:, 0]
        vec_y = m[i, :][:, 1]

        # vec_x = (m[i, :][:, 0] - dacadc_settings['ch1_offset'])
        # vec_y = (m[i, :][:, 1] - dacadc_settings['ch2_offset'])

        # md and mn for future implementation
        md = mdn[i, :][:, 0]
        mn = mdn[i, :][:, 1]

        mask = np.logical_and(np.logical_and(vec_x <= X_MAX, vec_x >= X_MIN),
                              np.logical_and(vec_y <= Y_MAX, vec_y >= Y_MIN))

        if np.any(mask == True):
            start, stop = np.where(mask == True)[0][0], np.where(mask == True)[0][-1]

            start = start.item()
            stop = stop.item()

            num_points = stop - start + 1

        # Ramping to initial values
        if i == 0:
            dac_adc.delay_unit(0)
            d_read = dac_adc.ramp1(dac1_ch, 0, vec_x[start], vgate2start_pnts, vgate2start_delay)
            time.sleep(1)
        else:
            dac_adc.delay_unit(0)
            previous_vec_x = m[i-1, :][:, 0] 
            d_read = dac_adc.ramp1(dac1_ch, previous_vec_x[stop], vec_x[start], vgate2next_pnts, vgate2next_delay)

        if i == 0:
            dac_adc.delay_unit(0)
            d_read = dac_adc.ramp1(dac2_ch, 0, vec_y[start], vset2start_pnts, vset2start_delay)
            time.sleep(1)
        else:
            dac_adc.delay_unit(0)
            previous_vec_y = m[i-1, :][:, 1] 
            d_read = dac_adc.ramp1(dac2_ch, previous_vec_y[stop], vec_y[start], vset2next_pnts, vset2next_delay)

        dac_adc.delay_unit(delay_unit)

        print("{} of {}  --> Ramping. Points: {}".format(i + 1, num_y, num_points))
        d_read = dac_adc.buffer_ramp([dac1_ch, dac2_ch],
                                     [adc1_ch, adc2_ch],
                                     [vec_x[start], vec_y[start]],
                                     [vec_x[stop], vec_y[stop]], num_points,
                                     delay_meas, ADC_AVGSIZE)

        d_tmp = d_read

        data_x[start:stop + 1], data_y[start:stop + 1] = d_tmp

        #radius = np.sqrt(np.square(data_x) + np.square(data_y)) * sens_var
        radius = np.array(data_x)
        phase = np.array(data_y)
        #phase = np.arctan2(data_y, data_x)

        # TODO rescale lock in sensitivity

        j = np.linspace(0, num_x - 1, num_x)
        ii = np.ones(num_x) * i
        t1 = np.ones(num_x) * time.time() - t0
        totdata = np.array([j, ii, vec_x, vec_y, radius, phase, md, mn, data_x, data_y, t1])
        dv.add(totdata.T)

        # Ramp down to zero if last point
        if (i == num_y-1):
            dac_adc.ramp1(dac1_ch, vec_x[stop], 0, 10000, 300)
            dac_adc.ramp1(dac2_ch, vec_x[stop], 0, 1000, 500)


    print("it took {} s. to write data".format(time.time() - t0))

if __name__ == '__main__':
    main()