import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import peakutils
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")


def dist_of_charging_vs_not():
    mean_energy_when_charging = np.zeros(electric_car_labels.shape[0])
    mean_energy_not_charging = np.zeros(electric_car_labels.shape[0])

    for n in xrange(electric_car_labels.shape[0]):
        indices = np.where(electric_car_labels[n,:]==1)[0]
        mean_energy_when_charging[n] = np.mean(electric_car_features[n,indices])
        off_indices = np.where(electric_car_labels[n,:]==0)[0]
        mean_energy_not_charging[n] = np.mean(electric_car_features[n,off_indices])

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

    text_string = "Energy $\Delta$ = {0}".format(np.round(x_plot[peak_indices[0]], 2))
    ax.text(x_plot[peak_indices[0]], kde_plot[peak_indices[0]] + 0.02, text_string,
            fontdict = {"ha": "center", "color": "r"})

    text_string = "Energy $\Delta$ = {0}".format(np.round(x_plot[peak_indices[1]], 2))
    ax.text(x_plot[peak_indices[1]], kde_plot[peak_indices[1]] + 0.02, text_string,
            fontdict = {"ha": "center", "color": "r"})

    ax.set_xticks(np.arange(0,5,0.2))
    ax.set_xlim(0.7,3.4)
    # ax.set_xlim(0.7,3.6)
    ax.set_xlabel("Average Increase in Energy Use While Charging", fontsize=20)

    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_ylim(0,1)
    ax.set_ylabel("Probablity Density", fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title("Difference in Energy Consumption While Charging", fontsize=25)

    ax.legend(fontsize=20, loc="upper left")
    plt.tight_layout()
    plt.savefig("images/eda/dist_of_charging_vs_not.png")
    fig.show()

def average_energy_use(days=None, save_figure=False):
    energy = np.mean(feature_matrix, axis=0)
    car_energy = np.mean(electric_car_features, axis=0)
    no_car_energy = np.mean(no_electric_car_features, axis=0)

    if days is None:
        days = [0, 59]

    na = 48*days[0]
    nb = 1 + 48*(days[1] + 1)

    if nb > len(energy):
        nb = len(energy)

    indices = np.arange(len(energy))
    time = 1.*indices/48

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    ax.plot(time[indices[na:nb]], energy[indices[na:nb]], label="All Houses")
    ax.plot(time[indices[na:nb]], car_energy[indices[na:nb]], label="Houses w/ Electric Car")
    ax.plot(time[indices[na:nb]], no_car_energy[indices[na:nb]], label="Houses w/o Electric Car")

    ax.set_xlabel("Time (Days)", fontsize=20)
    ax.set_xticks(np.arange(days[0], days[1]+2), minor=True)
    ax.xaxis.grid(True, which="minor")
    ax.xaxis.grid(False, which="major")
    ax.set_ylabel("Energy", fontsize=20)
    ax.set_title("Average Energy Use", fontsize=25)

    ax.tick_params(axis='both', which='major', labelsize=15)

    ax.legend(fontsize=20, ncol=3)

    plt.tight_layout()
    if save_figure:
        fig_name = "images/eda/average_energy_day{0}_to_day{1}.png".format(days[0]+1, days[1]+1)
        plt.savefig(fig_name)
    fig.show()

def single_house_energy_use(house, days=None, save_figure=False):
    house_id = features["House ID"][house].astype(int)
    energy = feature_matrix[house, :]
    charges = label_matrix[house, :]

    indices = np.arange(len(energy))
    time = 1.*indices/48
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

    ax.plot(time[indices[na:nb]], energy[na:nb], label="Energy Consumption")
    ax.plot(time[charge_ab], energy[charge_ab], 'o', label="Charge Points")

    ax.set_xlabel("Time (Days)", fontsize=20)
    ax.set_xticks(np.arange(days[0], days[1]+2), minor=True)
    ax.xaxis.grid(True, which="minor")
    ax.xaxis.grid(False, which="major")
    ax.set_ylabel("Energy", fontsize=20)
    ax.set_title("House {0} Energy Consumption".format(house_id), fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(fontsize=20)

    plt.tight_layout()
    if save_figure:
        fig_name = "images/eda/house_{0}_energy_day{1}_to_day{2}.png".format(house_id, days[0]+1, days[1]+1)
        plt.savefig(fig_name)
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

    # dist_of_charging_vs_not()
    # average_energy_use(days=[42,59], save_figure=True)
    # single_house_energy_use(houses_with_electric_cars[0], [0,20], save_figure=True)
    # single_house_energy_use(houses_with_electric_cars[0], [21,41], save_figure=True)
    # single_house_energy_use(houses_with_electric_cars[0], [42,59], save_figure=True)
