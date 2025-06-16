import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import shutil
import random
# Directory containing the data files

ac_data_dirs = ["AC_W3096/Nov272024", "AC_W3096/Dec42024", "AC_W3096/Dec52024", "AC_W3096/Dec62024", "AC_W3096/Dec92024", "AC_W3096/Dec122024", "AC_W3096/Dec112024"]
dc_data_dirs_3058 = ["DC_W3058/Oct252023", "DC_W3058/Oct262023", "DC_W3058/Oct272023", "DC_W3058/Oct302023", "DC_W3058/Oct312023", "DC_W3058/Nov012023", "DC_W3058/Nov022023", "DC_W3058/Nov032023"]
dc_data_dirs_3045 = ["DC_W3045/Sep212023", "DC_W3045/Sep252023", "DC_W3045/Sep262023", "DC_W3045/Sep282023", "DC_W3045/Oct022023"]
data_dirs = ac_data_dirs + dc_data_dirs_3058 + dc_data_dirs_3045
test_data_dirs = ["AC_W3096/Dec112024", "DC_W3058/Nov012023", "DC_W3045/Sep252023"]

def linear(x, m, b):
    return m*x + b

def linear_fit(x_data, y_data, p0, sigmas=None):
    if sigmas is None: 
        popt, pconv = curve_fit(linear, xdata = x_data , ydata = y_data, p0=p0)
    else: popt, pconv = curve_fit(linear, xdata = x_data , ydata = y_data, p0=p0, sigma=sigmas)
    perr= np.sqrt(np.diag(pconv))
    residual = np.linalg.norm(y_data-linear(x_data, *popt))
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (residual**2 / ss_tot)
    return popt, perr, residual, r_squared


def fit_breakdown(xs, ys, start_idx, dist_file=None, bd_thresh=0.5):
    # set up data structures
    results = {'random': {'color': 'black'}}
    rand_len = 100
    breakdown_vals = np.zeros(rand_len)
    pop0s = np.zeros(rand_len)
    pop1s = np.zeros(rand_len)
    # calculate linear fit for full range
    fit_range = int((len(xs)-start_idx)*0.5)
    fixed_popt, _, _, _ = linear_fit(xs[start_idx:start_idx+fit_range], ys[start_idx:start_idx+fit_range], p0=[1, -100])
    liny_vals = linear(xs[start_idx:], fixed_popt[0], fixed_popt[1])
    pop0s[0] = fixed_popt[0]
    pop1s[0] = fixed_popt[1]
    # calculate breakdown voltage
    for i, liny in enumerate(liny_vals):
            if abs(ys[i+start_idx] - liny) < bd_thresh:
                breakdown_vals[0] = xs[i+start_idx]
    # repeat rand_len times to use width for uncertainty calculation
    for rand_i in range(1, rand_len):
        indices = random.sample(range(start_idx, start_idx+fit_range), fit_range//2) # sample half the data
        fixed_popt, _, _, _ = linear_fit(xs[indices], ys[indices], p0=[1, -100])
        liny_vals = linear(xs[start_idx:], fixed_popt[0], fixed_popt[1])
        pop0s[rand_i] = fixed_popt[0]
        pop1s[rand_i] = fixed_popt[1]
        for i, liny in enumerate(liny_vals):
            if abs(ys[i+start_idx] - liny) < bd_thresh:
                breakdown_vals[rand_i] = xs[i+start_idx]
    # plot breakdown distribution
    plt.figure()
    plt.hist(breakdown_vals, histtype='step', bins=20)
    plt.ylabel('Frequency')
    plt.xlabel('Breakdown Voltage')
    plt.title('Breakdown Voltage Distribution From Random Ranges')
    plt.tight_layout()
    plt.savefig(dist_file)
    plt.clf()
    # saving breakdown and plotting information
    results['random']['x'] = xs[start_idx:]
    results['random']['y'] = linear(results['random']['x'], pop0s[0], pop1s[0])
    results['random']['bd'] = breakdown_vals[0] # breakdown voltage
    results['random']['bderr'] = max(np.std(breakdown_vals), 0.34) # breakdown uncertainty
    #print('breakdown at', int(results['random']['bd']), '+/-', float(int(results['random']['bderr']*1000))/1000)
    results['random']['bdi'] = len(xs)-start_idx-1 # index of breakdown
    for i, x in enumerate(xs[start_idx:]):
        if x > results['random']['bd']:
            results['random']['bdi'] = i
            break
    # keep plotting information for 10 of the random fits (100 would make plot look bad)
    for rand_i in range(10):
        results['random'+str(rand_i)] = dict()
        results['random'+str(rand_i)]['x'] = results['random']['x']
        results['random'+str(rand_i)]['y'] = linear(results['random']['x'], pop0s[rand_i], pop1s[rand_i])
        results['random'+str(rand_i)]['bd'] = breakdown_vals[rand_i]
        results['random'+str(rand_i)]['bderr'] = 2
        results['random'+str(rand_i)]['bdi'] = len(xs)-start_idx-1
        for i, x in enumerate(xs[start_idx:]):
            if x > results['random'+str(rand_i)]['bd']:
                results['random'+str(rand_i)]['bdi'] = i
                break
    return results

# Function to parse the metadata and data from a file
def parse_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Extract metadata
    if 'DC' in filepath: temperature = float(lines[0].split(': ')[1].split(' ')[0])
    else: temperature = float(lines[0].split(':')[1].strip().replace(" C", ""))
    date = lines[2].split(':', 1)[1].strip().split(' ')[0]
    # Extract table data
    data = np.genfromtxt(lines[4:], delimiter=',', names=['voltage','pad','gr','totalCurrent'])
    return temperature, date, data

# Map temperature to a rainbow color (purple to red)
def temperature_to_color(temperature):
    min_temp, max_temp = -60, 100
    norm_temp = (temperature - min_temp) / (max_temp - min_temp)
    return plt.cm.rainbow(norm_temp)

def humidity_to_color(rh):
    min_rh, max_rh = 0, 36
    norm_rh = (rh - min_rh) / (max_rh - min_rh)
    return plt.cm.rainbow(norm_rh)

def plot_scans(data_dir, curr_type, min_temp=0, bd_thresh=0.5):
    file_groups = {"ru_": [], "rd_": []}
    for filename in os.listdir(data_dir):
        if 'W3058' in data_dir and filename.endswith(".txt") and int(filename.split("_")[0]) > 0:
            key = "ru_" if int(filename.split("_")[0]) < 11 else "rd_"
            file_groups[key].append(os.path.join(data_dir, filename))
        elif 'W3045' in data_dir and filename.endswith(".txt") and int(filename.split("_")[0]) > 0:
            key = "rd_" if "Down" in filename else "ru_"
            file_groups[key].append(os.path.join(data_dir, filename))
        elif filename.endswith(".txt") and (filename.startswith("ru_") or filename.startswith("rd_")):
            key = "ru_" if filename.startswith("ru_") else "rd_"
            file_groups[key].append(os.path.join(data_dir, filename))
    # Loop through file groups and plot
    temp_dict = dict()
    bdv_dict = dict()
    bdverr_dict = dict()

    for ramp_type, files in file_groups.items():
        if files == []: continue
        plt.figure(figsize=(10, 6))
        plots = []  # To store plot data for sorting
        for filepath in sorted(files):
            temp, date, data = parse_file(filepath)
            if temp < min_temp and 'W' not in data_dir: continue
            color = temperature_to_color(temp)
            # Plot pad current and store data for later sorting
            neg_idx = data[curr_type] < 0
            if 'DC' in data_dir and curr_type != 'totalCurrent': 
                plots.append((abs(data['voltage'][~neg_idx]), data[curr_type][~neg_idx], temp, color))
            else: plots.append((abs(data['voltage'][neg_idx]), -1*data[curr_type][neg_idx], temp, color))
        # Sort by temperature (numerical order)
        plots.sort(key=lambda x: float(x[2]))  # Sorting by temperature
        # Plot each sorted line
        for volt, curr, temp, color in plots:
            plt.plot(volt, curr, label=str(temp)+" C", color=color, marker='o', markersize=3)
        # Plot configuration
        plt.xlabel("Voltage (V)")
        plt.ylabel(curr_type+" Current (A)")
        plt.yscale('log')
        plt.title(f"{date} - {'Pad' if curr_type == 'pad' else 'Guard Ring'} Current Temperature Scan ({'Ramp Up' if ramp_type == 'ru_' else 'Ramp Down'})")
        plt.legend(title="Temperature", loc='best')
        plt.grid(True)
        plt.tight_layout()
        # Save the plot
        plot_filename = data_dir+"/"+f"{ramp_type.strip('_')}_plot"+curr_type+".png"
        plt.savefig(plot_filename)
        plt.clf()
        if curr_type != 'pad': continue
        breakdown = dict()
        #sigma_breakdown = dict()
        if 'W3045' in data_dir: depletion=50
        else: depletion=36
        for volt, curr, temp, color in plots:
            if temp < min_temp and 'DC' not in data_dir: continue
            #elif temp == 60 and 'W3045' in data_dir and ramp_type == 'ru_' and date == '26/09/2023': continue
            #elif temp == 80 and 'W3045' in data_dir and ramp_type == 'ru_' and date == '26/09/2023': continue
            elif temp <= -20 and 'W3058' in data_dir and ramp_type == 'rd_' and date == '27/10/2023': continue
            elif temp == -60 and 'W3058' in data_dir and date == '27/10/2023': continue
            elif temp == -40 and 'W3058' in data_dir and ramp_type == 'rd_' and date == '26/10/2023': continue
            elif temp == -60 and 'W3058' in data_dir and date == '26/10/2023': continue
            elif temp < -20 and 'W3058' in data_dir and ramp_type == 'rd_' and date == '25/10/2023': continue
            elif temp < -20 and 'W3058' in data_dir and date == '30/10/2023': continue
            log_curr = np.log10(curr)
            # identify start point for baseline regression
            if temp == -20 and 'W3058' in data_dir and ramp_type == 'rd_' and date == '27/10/2023': temp_depletion = 66
            elif temp == -20 and 'W3058' in data_dir and ramp_type == 'rd_' and date == '30/10/2023': temp_depletion = 70
            else: temp_depletion = depletion
            for v_idx, voltage in enumerate(volt):
                if voltage >= temp_depletion:
                    base_start_idx = v_idx
                    break
            print('performing fit for', date, ramp_type, temp)
            plot_filename = data_dir+"/"+f"{ramp_type.strip('_')}_"+str(temp)+"_breakdown.png"
            dist_file = data_dir+"/"+f"{ramp_type.strip('_')}_"+str(temp)+"_bd_dist.png"
            # select treshold above fit that denotes breakdown
            if not isinstance(bd_thresh, float):
                if 'W3058' in data_dir: sensor_threshold = bd_thresh['W3058']
                elif 'W3045' in data_dir: sensor_threshold = bd_thresh['W3045']
                else: sensor_threshold = bd_thresh['AC']
            else: sensor_threshold = bd_thresh
            # fit the breakdown
            results = fit_breakdown(volt, log_curr, base_start_idx, dist_file=dist_file, bd_thresh=sensor_threshold)
            breakdown[temp] = dict()
            # plot
            plt.figure(figsize=(10, 6))
            plt.plot(volt[1:], log_curr[1:], label=f"{temp} C", color=color, marker='o', markersize=3)
            plotted_random = False
            for key, result in results.items():
                if key == 'random':
                    # store breakdown voltage and uncertainty for further analysis
                    breakdown[temp][key] = result['bd']
                    breakdown[temp][key+'err'] = result['bderr']
                    # plot actual breakdown fit
                    plt.plot(result['x'], result['y'], color=result['color'], linestyle='--', label='linear fit')
                    plt.plot(result['x'], result['y']+sensor_threshold, color='brown', linestyle='-.', label='threshold')
                    plt.scatter([result['bd']], [result['y'][result['bdi']]], color=result['color'], marker='*')
                else: # plot some of the random samples to visualize
                    if plotted_random: # plot without label
                        plt.plot(result['x'], result['y'], color='purple', linestyle=':')
                    else: # plot with label once
                        plt.plot(result['x'], result['y'], color='purple', linestyle=':', label='random samples')
                        plotted_random = True
                    plt.scatter([result['bd']], [result['y'][result['bdi']]], color='purple', marker='*')
                #if 'exp' in key: continue
            plt.xlabel("Voltage (V)")
            plt.ylabel("log(Pad Current (A))")
            plt.title("IV Scan with breakdown for "+str(temp)+" degrees: "+str(int(breakdown[temp]['random']))+" +/- "+str(float(int(10*breakdown[temp]['randomerr']))/10)+" V")
            valid_ylims = log_curr[1:][np.isfinite(log_curr[1:])]
            plt.ylim(np.min(valid_ylims)-.5, np.max(valid_ylims)+.5)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.clf()
        # get aggregate data for plotting
        plt.figure(figsize=(10, 6))
        temperatures, breakdown_vs, breakdown_errs = [], [], []
        for temp, bd_voltage in breakdown.items():
            temperatures.append(temp)
            breakdown_vs.append(bd_voltage['random'])
            breakdown_errs.append(bd_voltage['randomerr'])
        temperatures = np.array(temperatures)
        breakdown_vs = np.array(breakdown_vs)
        #sig_breakdown_vs = np.array(sig_breakdown_vs)
        popt, perr, _, r2 = linear_fit(temperatures, breakdown_vs, [1,150], sigmas=breakdown_errs)
        #plt.scatter(temperatures, breakdown_vs, label=key, color="purple")
        plt.errorbar(temperatures, breakdown_vs, yerr=breakdown_errs, fmt='o', capsize=5, label=key, color="black")
        plt.plot(temperatures, linear(temperatures, popt[0], popt[1]), label=f'r2 value: {r2:0.2f} Best Fit: y = {popt[0]:.2f}x + { popt[1]:.2f}', color='purple', linestyle='--')
        plt.xlabel("Temperature (C)")
        plt.ylabel("Breakdown Voltage (V)")
        plt.title("Breakdown Voltage as a Function of Temperature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(data_dir+"/"+f"{ramp_type.strip('_')}_fit_breakdown.png")
        plt.clf()
        # return fits so that they can be plotted together
        temp_dict[ramp_type] = temperatures
        bdv_dict[ramp_type] = breakdown_vs
        bdverr_dict[ramp_type] = breakdown_errs
    return temp_dict, bdv_dict, bdverr_dict

def plot_sensor(data_dirs, temps, breakdown_volts, breakdown_sigs, sensor):
    # plot all scans together for a given sensor
    plt.figure(figsize=(10, 6))
    means = dict()
    sigmas = dict()
    # get the data and store in dictionary
    for i, data_dir in enumerate(data_dirs):
        print(sensor, data_dir)
        if sensor == 'AC_3096': 
            if 'DC' in data_dir: continue
        elif sensor not in data_dir: continue
        for ramp_type, ramp_temp in temps[i].items():
            #if ramp_type == 'rd_': continue
            ramp_bdv = breakdown_volts[i][ramp_type]
            ramp_bdsig = breakdown_sigs[i][ramp_type]
            # add scan to plot
            plt.plot(ramp_temp, ramp_bdv, marker='o', markersize=3, label=data_dir)
            # add each individual measurement to the dictionaries
            for j, temp in enumerate(ramp_temp):
                if temp in means:
                    means[temp].append(ramp_bdv[j])
                    sigmas[temp].append(ramp_bdsig[j])
                else:
                    means[temp] = [ramp_bdv[j]]
                    sigmas[temp] = [ramp_bdsig[j]]
    # plot formatting
    plt.xlabel("Temperature (C)")
    plt.ylabel("Breakdown Voltage (V)")
    plt.title(sensor+" Breakdown Voltage as a Function of Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/all_breakdown_fits_"+sensor+".png")
    plt.clf()
    # plot breakdown voltages for each temp as a function of scan number
    plt.figure(figsize=(10, 6))

    for temp, bdvs in means.items():
        if temp == -20 and sensor == 'DC_W3058': scan_idx = np.array([0,1,2,3,4,6,7,8,9,10,11,12,13,14,15])
        elif temp == -40 and sensor == 'DC_W3058': scan_idx = np.array([0,2,4,8,9,10,11,12,13,14,15])
        elif temp == -60 and sensor == 'DC_W3058': scan_idx = np.array([0,8,9,10,11,12,13,14,15])
        elif temp == 80 and sensor == 'DC_W3058': scan_idx = np.array([0,1,2,3,5,6,7,8,9,10,11,12,13,14,15])
        elif temp == 40 and sensor == 'DC_W3058': scan_idx = np.array([0,1,2,3,4,5,6,7,9,10,11,12,13,14,15])
        else: scan_idx = np.arange(len(bdvs))
        plt.plot(scan_idx, bdvs, marker='o', color=temperature_to_color(temp), label=str(temp)+" C")
    plt.xlabel("Scan Number")
    plt.ylabel("Breakdown Voltage (V)")
    plt.title(sensor+" Breakdown Voltage by Temperature over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/breakdown_over_time_"+sensor+".png")
    plt.clf()
    # calculate weighted mean and its uncertainty across measurements
    avg_sigma = 0
    num_sigmas = 0
    fit_temps, fit_bds, fit_sigs = [], [], []
    for temp, sigma in sigmas.items():
        sigma = np.array(sigma)
        fit_temps.append(temp)
        fit_sigs.append(np.sqrt(1/np.sum(1/(sigma**2))))
        fit_bds.append(np.sum(np.array(means[temp])/(sigma**2))*(fit_sigs[-1]**2))
        #fit_bds.append(np.sum(np.array(means[temp])/(np.array(sigma)**2)))/(fit_sigs[-1]**2)
        print(fit_bds[-1], fit_sigs[-1])
        avg_sigma += fit_sigs[-1]
        num_sigmas += 1
    if num_sigmas > 0: avg_sigma /= num_sigmas
    print(sensor+" AVERAGE UNCERTAINTY", avg_sigma)
    # calculate and plot overall fit
    plt.figure(figsize=(10, 6))
    fit_temps = np.array(fit_temps)
    fit_bds = np.array(fit_bds)
    fit_sigs = np.array(fit_sigs)
    popt, perr, _, r2 = linear_fit(fit_temps, fit_bds, [1,150], sigmas=fit_sigs)
    plt.errorbar(fit_temps, fit_bds, yerr=fit_sigs, fmt='o', capsize=5, label=sensor+' data')
    plt.plot(fit_temps, linear(fit_temps, popt[0], popt[1]), label=f'r2 value: {r2:0.2f} Best Fit: y = {popt[0]:.2f}x + { popt[1]:.2f}', linestyle='--')
    slope_err = perr[0]
    print(sensor+" SLOPE UNCERTAINTY", slope_err)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Breakdown Voltage (V)")
    plt.title(sensor+" Breakdown Voltage as a Function of Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/averaged_fit_"+sensor+".png")
    plt.clf()
    #return avg_sigma
    return slope_err, avg_sigma

def plot_all(data_dirs, max_res=0.01, min_temp=0, bd_thresh=0.5):
    temps, breakdown_volts, breakdown_sigs = [], [], []
    for data_directory in data_dirs:
        data_dir = 'data/'+data_directory
        temp, bdv, bdverr = plot_scans(data_dir, 'pad', min_temp=min_temp, bd_thresh=bd_thresh)
        temps.append(temp)
        breakdown_volts.append(bdv)
        breakdown_sigs.append(bdverr)
        plot_scans(data_dir, 'gr')
        plot_scans(data_dir, 'totalCurrent')
    ac_slope_err, ac_avg_sigma = plot_sensor(data_dirs, temps, breakdown_volts, breakdown_sigs, sensor='AC_3096')
    w3058_slope_err, w3058_avg_sigma = plot_sensor(data_dirs, temps, breakdown_volts, breakdown_sigs, sensor='DC_W3058')
    w3045_slope_err, w3045_avg_sigma = plot_sensor(data_dirs, temps, breakdown_volts, breakdown_sigs, sensor='DC_W3045')
    return [ac_slope_err, w3058_slope_err, w3045_slope_err]#, [ac_avg_sigma, w3058_avg_sigma, w3045_avg_sigma]

def plot_humidity_scans(data_dir, bd_thresh):
    curr_type = 'pad'
    sensor='AC'
    plt.figure(figsize=(10, 6))
    humidities = []
    filenames = []
    for filename in os.listdir(data_dir):
        if filename.endswith("p__21.txt") and filename.startswith("rh_"):
            filenames.append(filename)
            humidities.append(float(filename.split('_')[1][:-1]))
    leakage_140 = []
    for i in np.argsort(humidities):
        _, _, data = parse_file(os.path.join(data_dir, filenames[i]))
        neg_idx = data[curr_type] < 0
        voltages = abs(data['voltage'][neg_idx])
        log_curr = np.log10(-1*data[curr_type][neg_idx])
        plt.scatter(voltages, log_curr, color=humidity_to_color(humidities[i]), label=str(humidities[i])+' % rh', s=20)
        leakage_140.append(log_curr[np.argmin(abs(voltages-140))])
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("log( pad current (A) )")
    plt.title(sensor+"-LGAD IV Scan as Function of Relative Humidity at 21 C")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("data/humidity_"+sensor+".png")
    plt.clf()
    # plot leakage current at 140V (approximate operating voltage)
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(humidities), leakage_140, marker='o')
    plt.xlabel("Relative Humidity (%)")
    plt.ylabel("log( pad current (A) ) @ 140V")
    plt.title(sensor+"-LGAD Leakage Current at 140V as Function of Relative Humidity at 21 C")
    plt.tight_layout()
    plt.savefig("data/humidity_leakage_"+sensor+".png")
    plt.clf()


def find_threshold(data_dirs, min_temp, bd_thresholds):
    min_keys = ["AC_W3096", "DC_W3058", "DC_W3045"]
    min_sigma = {"AC_W3096": 999, "DC_W3058": 999, "DC_W3045": 999}
    min_threshold = {"AC_W3096": 0, "DC_W3058": 0, "DC_W3045": 0}
    for thresh in bd_thresholds:
        sigmas = plot_all(data_dirs, min_temp=min_temp, bd_thresh=thresh)
        for i, sigma in enumerate(sigmas):
            if sigma < min_sigma[min_keys[i]]: 
                min_sigma[min_keys[i]] = sigma
                min_threshold[min_keys[i]] = thresh
    print(min_threshold)
    print(min_sigma)
    return min_threshold

def run_full(data_dirs, min_temp):
    #thresholds = find_threshold(data_dirs, min_temp, bd_thresholds=np.linspace(.1,0.6,6))
    thresholds = {"AC_W3096": 0.3, "DC_W3058": 0.6, "W3045": 0.4}
    plot_all(data_dirs, min_temp=min_temp, bd_thresh=thresholds)
    plot_humidity_scans("data/AC_W3096/Dec102024", thresholds["AC_W3096"])

#find_threshold(test_data_dirs, min_temp=60, bd_thresholds=[{"AC": 0.2, "W3058": 0.2, "W3045": 0.2}, 0.5, 0.7])
run_full(data_dirs, min_temp=0)
#run_full(test_data_dirs, min_temp=0)
    
