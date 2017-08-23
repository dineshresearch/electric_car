import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import peakutils
from sklearn.preprocessing import StandardScaler
import calendar
from sklearn.neighbors import NearestNeighbors

sns.set_style("whitegrid")


def dist_of_charging_vs_not():
    '''
    INPUT:
        - x: x-coordinates (ndarray)
        - y: y-coordinates (ndarray)
        - pad: number of data points to pad the x, y vectors with (int)
        - thresh: threshold used for noise removal (float)
    OUTPUT: None
    '''
    mean_energy_when_charging = np.zeros(electric_car_labels.shape[0])
    mean_energy_not_charging = np.zeros(electric_car_labels.shape[0])

    for n in xrange(electric_car_labels.shape[0]):
        increments = np.where(electric_car_labels[n,:]==1)[0]
        mean_energy_when_charging[n] = np.mean(electric_car_features[n,increments])
        off_increments = np.where(electric_car_labels[n,:]==0)[0]
        mean_energy_not_charging[n] = np.mean(electric_car_features[n,off_increments])

    x_fit = (mean_energy_when_charging - np.mean(electric_car_features,axis=1))
    # x_fit = mean_energy_when_charging - mean_energy_not_charging
    x_plot = np.linspace(np.min(x_fit), np.max(x_fit), 1000)

    kde = gaussian_kde(x_fit)
    kde_plot = kde(x_plot)

    peak_indices = peakutils.indexes(kde_plot)

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    ax.hist(x_fit, bins=25, normed=True, alpha=0.5, label="Histogram")
    ax.plot(x_plot, kde_plot, label="KDE")
    ax.plot(x_plot[peak_indices], kde_plot[peak_indices], 'rX', label="Local Maxima")

    text_string = "Power $\Delta$ = {0}".format(np.round(x_plot[peak_indices[0]], 2))
    ax.text(x_plot[peak_indices[0]], kde_plot[peak_indices[0]] + 0.02, text_string,
            fontdict = {"ha": "center", "color": "r"})

    text_string = "Power $\Delta$ = {0}".format(np.round(x_plot[peak_indices[1]], 2))
    ax.text(x_plot[peak_indices[1]], kde_plot[peak_indices[1]] + 0.02, text_string,
            fontdict = {"ha": "center", "color": "r"})

    ax.set_xticks(np.arange(0,5,0.2))
    ax.set_xlim(0.7,3.4)
    # ax.set_xlim(0.7,3.6)
    ax.set_xlabel("Average Increase in Power Use While Charging", fontsize=20)

    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_ylim(0,1)
    ax.set_ylabel("Probablity Density", fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title("Difference in Power Usage While Charging", fontsize=25)

    ax.legend(fontsize=20, loc="upper left")
    plt.tight_layout()
    plt.savefig("images/eda/dist_of_charging_vs_not.png")
    fig.show()

def average_power_usage(days=None, save_figure=False, show_day_labels=False):
    '''
    INPUT:
        - x: x-coordinates (ndarray)
        - y: y-coordinates (ndarray)
        - pad: number of data points to pad the x, y vectors with (int)
        - thresh: threshold used for noise removal (float)
    OUTPUT: None
    '''
    energy = np.mean(feature_matrix, axis=0)
    car_energy = np.mean(electric_car_features, axis=0)
    no_car_energy = np.mean(no_electric_car_features, axis=0)

    if days is None:
        days = [0, 59]

    na = 48*days[0]
    nb = 1 + 48*(days[1] + 1)

    if nb > len(energy):
        nb = len(energy)

    increments = np.arange(len(energy))
    time = 1.*increments/48

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    ax.plot(time[increments[na:nb]], energy[increments[na:nb]], label="All Houses")
    ax.plot(time[increments[na:nb]], car_energy[increments[na:nb]], label="Houses w/ Electric Car")
    ax.plot(time[increments[na:nb]], no_car_energy[increments[na:nb]], label="Houses w/o Electric Car")

    if show_day_labels:
        xticks = np.arange(days[0], days[1] + 1)
        ax.set_xticks(0.5 + xticks)
        day_names = [list(calendar.day_abbr)[n] for n in (4 + xticks)%7]
        xlabels = ["{0} - {1}".format(1 + xticks[n], day_names[n]) for n in xrange(len(xticks))]
        ax.set_xticklabels(xlabels, rotation = 90)
        ax.set_xlim(days[0], days[1]+1)
        ax.set_xlabel("Time (Day Number - Day Of Week)", fontsize=20)
    else:
        ax.set_xlabel("Time (Days)", fontsize=20)
    ax.set_xticks(np.arange(days[0], days[1]+2), minor=True)
    ax.xaxis.grid(True, which="minor")
    ax.xaxis.grid(False, which="major")
    ax.set_ylabel("Power", fontsize=20)
    ax.set_title("Average Power Usage: Day {0} to Day {1}".format(days[0]+1, days[1]+1), fontsize=25)

    ax.tick_params(axis='both', which='major', labelsize=15)

    ax.legend(fontsize=20, ncol=3)

    plt.tight_layout()
    if save_figure:
        fig_name = "images/eda/average_power_day{0}_to_day{1}.png".format(days[0]+1, days[1]+1)
        plt.savefig(fig_name)
    fig.show()

def single_house_power_usage(house, days=None, save_figure=False, show_day_labels=False):
    '''
    INPUT:
        - x: x-coordinates (ndarray)
        - y: y-coordinates (ndarray)
        - pad: number of data points to pad the x, y vectors with (int)
        - thresh: threshold used for noise removal (float)
    OUTPUT: None
    '''
    house_id = features["House ID"][house].astype(int)
    energy = feature_matrix[house, :]
    charges = label_matrix[house, :]

    increments = np.arange(len(energy))
    time = 1.*increments/48
    charge_indices = np.where(charges==1)[0]

    if days is None:
        days = [0, 59]

    na = 48*days[0]
    nb = 1 + 48*(days[1] + 1)

    if nb > len(energy):
        nb = len(energy)

    charge_ab = charge_indices[(charge_indices>=na) & (charge_indices<nb)]

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    ax.plot(time[increments[na:nb]], energy[na:nb], label="Power Usage")
    ax.plot(time[charge_ab], energy[charge_ab], 'o', label="Charge Points")

    if show_day_labels:
        xticks = np.arange(days[0], days[1] + 1)
        ax.set_xticks(0.5 + xticks)
        day_names = [list(calendar.day_abbr)[n] for n in (4 + xticks)%7]
        xlabels = ["{0} - {1}".format(1 + xticks[n], day_names[n]) for n in xrange(len(xticks))]
        ax.set_xticklabels(xlabels, rotation = 90)
        ax.set_xlim(days[0], days[1]+1)
        ax.set_xlabel("Time (Day Number - Day Of Week)", fontsize=20)
    else:
        ax.set_xlabel("Time (Days)", fontsize=20)
    ax.set_xticks(np.arange(days[0], days[1]+2), minor=True)
    ax.xaxis.grid(True, which="minor")
    ax.xaxis.grid(False, which="major")
    ax.set_ylabel("Power", fontsize=20)
    ax.set_title("House {0} Power Usage: Day {1} to Day {2}".format(house_id, days[0]+1, days[1]+1), fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(fontsize=20)

    plt.tight_layout()
    if save_figure:
        fig_name = "images/eda/house_{0}_power_day{1}_to_day{2}.png".format(house_id, days[0]+1, days[1]+1)
        plt.savefig(fig_name)
    fig.show()

def compare_ave_power_use_to_cars_charging():
    '''
    INPUT:
        - x: x-coordinates (ndarray)
        - y: y-coordinates (ndarray)
        - pad: number of data points to pad the x, y vectors with (int)
        - thresh: threshold used for noise removal (float)
    OUTPUT: None
    '''
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(111)

    increments = np.arange(feature_matrix.shape[1])
    time = increments/48.
    ax1.plot(time, np.mean(feature_matrix, axis=0), color="blue")
    ax1.set_xlabel("Time (Days)", fontsize=20, color="black")
    ax1.set_ylabel("Average Power Usage", fontsize=20, color="blue")
    ax1.tick_params(axis='y', which='major', labelsize=15, colors="blue")
    ax1.tick_params(axis='x', which='major', labelsize=15, colors="black")
    ax1.yaxis.grid(False, which="both")
    ax1.set_xticks(np.arange(0, 61), minor=True)
    ax1.xaxis.grid(True, which="minor")
    ax1.xaxis.grid(False, which="major")
    ax1.set_xlim(0,60)

    ax2 = ax1.twinx()
    ax2.plot(time, np.sum(label_matrix, axis=0), color="red", alpha=0.5)
    ax2.set_ylabel("Number of Cars Charging", fontsize=20, color="red")
    ax2.tick_params(axis='y', which='major', labelsize=15, colors="red")
    ax2.yaxis.grid(False, which="both")
    ax2.set_xlim(0,60)

    ax1.set_title("Average Power Usage Compared to Number of Cars Charging", fontsize=25)

    plt.tight_layout()
    plt.savefig("images/eda/average_power_cars_charging.png")
    fig.show()

if __name__ == '__main__':
    features = pd.read_csv("data/EV_train.csv")
    labels = pd.read_csv("data/EV_train_labels.csv")

    features = features.T.fillna(features[features.columns[1:]].T.median()).T

    feature_matrix = features[features.columns[1:]].values
    scaled_feature_matrix = StandardScaler().fit_transform(feature_matrix.T).T
    label_matrix = labels[labels.columns[1:]].values

    total_energy_per_house = np.sum(feature_matrix, axis=1)
    total_charges_per_house = np.sum(label_matrix, axis=1)

    houses_with_electric_cars = np.where(total_charges_per_house>0)[0]
    houses_without_electric_cars = np.where(total_charges_per_house==0)[0]

    electric_car_features = feature_matrix[houses_with_electric_cars, :]
    no_electric_car_features = feature_matrix[houses_without_electric_cars, :]
    electric_car_labels = label_matrix[houses_with_electric_cars, :]
    no_electric_car_labels = label_matrix[houses_without_electric_cars, :]
    electric_car_average = np.mean(electric_car_features,axis=0)
    no_electric_car_average = np.mean(no_electric_car_features,axis=0)
    all_average = np.mean(feature_matrix,axis=0)

    car_neighbors = NearestNeighbors(n_neighbors = 5)
    car_neighbors.fit(electric_car_features)
    car_nearest = car_neighbors.kneighbors(electric_car_average.reshape(1,-1))

    no_car_neighbors = NearestNeighbors(n_neighbors = 5)
    no_car_neighbors.fit(no_electric_car_features)
    no_car_nearest = no_car_neighbors.kneighbors(no_electric_car_average.reshape(1,-1))

    neighbors = NearestNeighbors(n_neighbors = 5)
    neighbors.fit(feature_matrix)
    nearest = neighbors.kneighbors(all_average.reshape(1,-1))
    # car_nearest = neighbors.kneighbors(electric_car_average.reshape(1,-1))
    # no_car_nearest = neighbors.kneighbors(no_electric_car_average.reshape(1,-1))

    # dist_of_charging_vs_not()
    # compare_ave_power_use_to_cars_charging()
    # average_power_usage(days=None, save_figure=True)
    # average_power_usage(days=[0,20], save_figure=False, show_day_labels=True)
    # average_power_usage(days=[21,41], save_figure=True, show_day_labels=True)
    # average_power_usage(days=[42,59], save_figure=True, show_day_labels=True)

    # single_house_power_usage(houses_with_electric_cars[390], [0,20], save_figure=False, show_day_labels=True)
    # single_house_power_usage(houses_with_electric_cars[5], [0,20], save_figure=False, show_day_labels=True)
    # single_house_power_usage(houses_with_electric_cars[3], [0,20], save_figure=False, show_day_labels=True)
    # single_house_power_usage(houses_with_electric_cars[2], [21,41], save_figure=False)
    # single_house_power_usage(houses_with_electric_cars[2], [42,59], save_figure=False)
