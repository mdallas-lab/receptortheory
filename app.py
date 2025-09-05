
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

st.title("Receptor Theory Dashboard")

# Load data
df_ach = pd.read_excel("PM2MDT_Ileum_Student_Data.xlsx", sheet_name="Acetylcholine Data", engine="openpyxl", skiprows=2, nrows=10)
df_atropine = pd.read_excel("PM2MDT_Ileum_Student_Data.xlsx", sheet_name="Atropine", engine="openpyxl", skiprows=3, nrows=10)

def prepare_data(df):
    concentrations = df.iloc[:, 0].astype(float)
    responses = df.iloc[:, 1:6].astype(float)
    mean_response = responses.mean(axis=1)
    return concentrations, mean_response

def sigmoid(x, bottom, top, logEC50, hill_slope):
    return bottom + (top - bottom) / (1 + 10**((logEC50 - np.log10(x)) * hill_slope))

def fit_curve(conc, resp):
    popt, _ = curve_fit(sigmoid, conc, resp, bounds=([0, 0, -10, 0.1], [1000, 1000, 0, 5]))
    return popt

def plot_curve(conc, resp, label):
    popt = fit_curve(conc, resp)
    x_fit = np.logspace(np.log10(min(conc)), np.log10(max(conc)), 100)
    y_fit = sigmoid(x_fit, *popt)
    plt.plot(x_fit, y_fit, label=f"{label} (EC50={10**popt[2]:.2e})")
    plt.scatter(conc, resp)
    plt.xscale('log')
    plt.xlabel("Concentration (M)")
    plt.ylabel("Response (mN)")
    plt.title("Concentration-Response Curve")
    plt.legend()

def schild_plot(df_control, df_antagonist):
    conc_control, resp_control = prepare_data(df_control)
    conc_ant, resp_ant = prepare_data(df_antagonist)
    popt_control = fit_curve(conc_control, resp_control)
    popt_ant = fit_curve(conc_ant, resp_ant)
    ec50_control = 10**popt_control[2]
    ec50_ant = 10**popt_ant[2]
    dose_ratio = ec50_ant / ec50_control
    log_dr_minus1 = np.log10(dose_ratio - 1)
    antagonist_conc = 1e-7  # 100 nM
    log_antagonist_conc = np.log10(antagonist_conc)
    plt.figure()
    plt.scatter(log_antagonist_conc, log_dr_minus1)
    plt.plot([log_antagonist_conc - 1, log_antagonist_conc + 1], [log_dr_minus1, log_dr_minus1], linestyle='--')
    plt.xlabel("Log[Antagonist]")
    plt.ylabel("Log(Dose Ratio - 1)")
    plt.title("Schild Plot")
    st.pyplot(plt)

# Prepare data
conc_ach, resp_ach = prepare_data(df_ach)
conc_atropine, resp_atropine = prepare_data(df_atropine)

# Plot concentration-response curves
plt.figure()
plot_curve(conc_ach, resp_ach, "Control")
plot_curve(conc_atropine, resp_atropine, "Atropine")
st.pyplot(plt)

# Plot Schild plot
schild_plot(df_ach, df_atropine)
