#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:00:36 2019

@author: snagappa
"""


import io
from urllib.request import urlopen
import urllib.parse
import json
import dill as pickle
import pandas as pd
from geopy.geocoders import Nominatim
import numpy as np
from scipy.interpolate import NearestNDInterpolator

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


_EPS = np.finfo(np.float).eps


def generate_latlon(df_referendum):
    """Add latitude and longitude to referendum data by querying Nominatim"""
    # geolocator = Nominatim(user_agent="france_latlon")
    commune_names = df_referendum["Libellé de la commune"]
    commune_location = {}
    for ix_, commune_ in enumerate(commune_names.values):
        try:
            # location = geolocator.geocode(
            #     {"city": commune_, "country": "France"})
            location = gmaps_geocode(commune_, "France")
        except:
            location = None
            err_str = " LOCATION NOT FOUND"
        else:
            err_str = ""
        print(ix_, ': ', commune_, err_str)
        commune_location[commune_] = location

        if not (ix_ % 10):
            print(ix_, ": saving")
            with io.open("latlon.p", 'wb') as f:
                pickle.dump(commune_location, f)
    with io.open("latlon.p", 'wb') as f:
        pickle.dump(commune_location, f)
    return commune_location


def _fix_url(url):
    """Convert UTF-8 strings to ascii format for use with urlopen()"""
    url = urllib.parse.urlsplit(url)
    url = list(url)
    url[2] = urllib.parse.quote(url[2])
    url = urllib.parse.urlunsplit(url)
    url.replace("%3D", '=')
    url.replace("%26", '&')
    return url


def gmaps_geocode(*args):
    """Returns the first result from querying the Google Maps geocoding API"""
    search_args = '+'.join(args)
    api_key = "AIzaSyDnxVKbabZD_-D2Xu1cgahETX50phYzrxI"
    url = ('https://maps.googleapis.com/maps/api/geocode/json?address='
           + _fix_url(search_args) + '&key='+api_key)
    jsonurl = urlopen(url)

    text = json.loads(jsonurl.read())
    return text["results"][0]
#    print ( text['results'][0]["formatted_address"])
#    print ( text['results'][0]["geometry"]['location']["lat"])
#    print ( text['results'][0]["geometry"]['location']["lng"])


def read_referendum_data():
    """Load the reference data"""
    referendum_file = "../data/Referendum.csv"
    df_referendum = pd.read_csv(referendum_file, delimiter=';')
    # Convert department code to str
    df_referendum["Code du département"] = (
        df_referendum["Code du département"].astype(str))
    return df_referendum


def get_referendum_by_regions():
    """Compute aggregate results from mainland, overseas territories and
    abroad."""
    drop_cols = ['Code du département', 'Libellé du département',
                 'Code de la commune', 'Libellé de la commune']

    def get_summary(df_list):
        df = pd.concat(df_list).drop(columns=drop_cols).sum()
        df = df/df["Inscrits"]*100
        df.drop("Inscrits", inplace=True)
        return df

    df_referendum = read_referendum_data()
    dfgroup = list(df_referendum.groupby("Code du département"))

    depcode = [dfg_[0] for dfg_ in dfgroup]
    mainland = get_summary(
        [dfgroup[ix_][1] for ix_ in range(len(depcode))
         if 'Z' not in depcode[ix_]])
    overseasterr = get_summary(
        [dfgroup[ix_][1] for ix_ in range(len(depcode))
         if 'Z' in depcode[ix_]])
    abroad = get_summary([dfgroup[-1][1]])
    return mainland, overseasterr, abroad


def read_latlon_data():
    """Load France mainland lat-lon data
    https://freakonometrics.hypotheses.org/1125
    """
    latlon_file = "../data/pop_fr_geoloc_1975_2010/pop_fr_geoloc_1975_2010.csv"
    df_latlon = pd.read_csv(
        latlon_file,
        usecols=["Code du département", "Code de la commune",
                 "Longitude", "Latitude"])
    # Convert department code to str
    df_latlon["Code du département"] = (
        df_latlon["Code du département"].astype(str))
    return df_latlon


def get_missing_latlon(df):
    """Make a Google Maps geocoding query for locations without a lat-lon"""
    try:
        with io.open("pickle/nogeoloc_list.p", 'rb') as f:
            latlon_data = pickle.load(f)
    except:
        latlon_data = {}
    # Indices without geolocation
    nogeoloc_ix = np.where(np.isnan(df["Latitude"]))
    # Rows from the DF corresponding to missing geolocation
    nogeoloc_list = df.iloc[nogeoloc_ix]

    # Iterate over missing rows
    for index_, row_ in nogeoloc_list.iterrows():
        # print(index_, row_["Libellé de la commune"],
        #       row_["Libellé du département"])
        # Make a query if necessary
        if (index_ not in latlon_data or
                np.isnan(latlon_data[index_]["geometry"]["location"]["lat"])):
            result = gmaps_geocode(row_["Libellé de la commune"],
                                   row_["Libellé du département"])
            latlon_data[index_] = result
    # Store results for loading later
    with io.open("pickle/nogeoloc_list.p", 'wb') as f:
        pickle.dump(latlon_data, f)

    # Overwrite lat-lon columns in the DF using the updated values
    latitude = np.asarray(df["Latitude"])
    longitude = np.asarray(df["Longitude"])
    new_lat = [latlon_data[index_]["geometry"]["location"]["lat"]
               for index_ in sorted(latlon_data)]
    new_lon = [latlon_data[index_]["geometry"]["location"]["lng"]
               for index_ in sorted(latlon_data)]
    latitude[nogeoloc_ix] = new_lat
    longitude[nogeoloc_ix] = new_lon
    df["Latitude"] = latitude
    df["Longitude"] = longitude


def read_popstruct_data():
    pfile = "pickle/popstructure_com2013.p"
    try:
        df_popstr = pd.read_pickle(pfile)
    except FileNotFoundError:
        df_popstr = pd.read_excel(
            "../data/popstructure_com2013.xls",
            sheet_name="COM_2013",
            index_col=None)
        df_popstr.to_pickle(pfile)
    geocode = df_popstr["Code géographique"].values
    depcode = df_popstr["Département"].values
    comcode = [int(gcode_[len(dcode_):])
               for gcode_, dcode_ in zip(geocode, depcode)]
    # Strip leading zeros from the depcode to match referendum data
    depcode = [dcode_.lstrip('0') for dcode_ in depcode]
    cols = df_popstr.keys()[5:]
    df_popstr = df_popstr[cols]
    # Convert all numbers to percent of total population
    # total_population = df_popstr["Population en 2013 (princ)"]
    # df_popstr = df_popstr.div(total_population, axis=0)
    # df_popstr["Population en 2013 (princ)"] = total_population
    df_popstr["Code du département"] = depcode
    df_popstr["Code de la commune"] = comcode
    return df_popstr


def read_popeducation_data():
    pfile = "pickle/popeducation_com2013.p"
    try:
        df_popedu = pd.read_pickle(pfile)
    except FileNotFoundError:
        df_popedu = pd.read_excel(
            "../data/education/pop-16ans-dipl6815_com2013.xls",
            sheet_name="COM_2010",
            index_col=None)
        df_popedu.to_pickle(pfile)
    depcode = df_popedu['Département\nen géographie courante'].values
    comcode = df_popedu['Commune\nen géographie courante'].values
    # Strip leading zeros from the depcode to match referendum data
    depcode = [dcode_.lstrip('0') for dcode_ in depcode]

    cols = df_popedu.keys()[6:]
    df_popedu = df_popedu[cols]
    df_popedu["Code du département"] = depcode
    df_popedu["Code de la commune"] = comcode
    return df_popedu


def read_popfiscal_data():
    pfile = "pickle/popfiscal_com2013.p"
    try:
        df_popfiscal = pd.read_pickle(pfile)
    except FileNotFoundError:
        df_popfiscal = pd.read_excel(
            "../data/fiscal/fiscal_com2013.xls",
            sheet_name="COM",
            index_col=None)
        df_popfiscal.to_pickle(pfile)
    geocode = df_popfiscal["Code géographique"]
    depcode, comcode = zip(*[(gcode_[:2], gcode_[2:]) for gcode_ in geocode])
    # Strip leading zeros from the depcode to match referendum data
    depcode = [dcode_.lstrip('0') for dcode_ in depcode]
    comcode = list(map(int, [ccode_.lstrip('0') for ccode_ in comcode]))

    cols = df_popfiscal.keys()[2:]
    df_popfiscal = df_popfiscal[cols]
    df_popfiscal["Code du département"] = depcode
    df_popfiscal["Code de la commune"] = comcode
    return df_popfiscal


def preprocess(df_referendum, df_latlon, df_popstr, df_popedu, df_popfiscal,
               fill_nn=True):
    # Normalise the referendum data to percentage of registered votes
    # total_votes = df_referendum["Choix A"] + df_referendum["Choix B"] + _EPS
    df_referendum["Choix A (%)"] = (
        df_referendum["Choix A"]/df_referendum["Inscrits"]*100)
    df_referendum["Choix B (%)"] = (
        df_referendum["Choix B"]/df_referendum["Inscrits"]*100)
    df_referendum["Abstentions (%)"] = (
        df_referendum["Abstentions"]/df_referendum["Inscrits"]*100)
    df_referendum["Blancs (%)"] = (
        df_referendum["Blancs et nuls"]/df_referendum["Inscrits"]*100)

    # Merge latitude and longitude with the referendum data
    df_latlon = pd.merge(df_referendum, df_latlon, how="left")

    # Use geocoding to get lat-lon of places not in the latlon file
    print("Adding new lat-lon data")
    get_missing_latlon(df_latlon)

    # Merge DFs
    df_pop_str_edu = pd.merge(df_popstr, df_popedu, how="outer")
    # Remove the department and commune codes prior to normalisation
    dep_com_labels = ["Code du département", "Code de la commune"]
    dep_com_code = df_pop_str_edu[dep_com_labels]
    df_pop_str_edu.drop(columns=dep_com_labels, inplace=True)
    # Normalise by total population, convert all numbers to % of total pop.
    total_population = df_pop_str_edu["Population en 2013 (princ)"]
    df_pop_str_edu = df_pop_str_edu.div(total_population + _EPS, axis=0)
    df_pop_str_edu["Population en 2013 (princ)"] = total_population
    df_pop_str_edu[dep_com_labels] = dep_com_code

    df = pd.merge(
        pd.merge(df_latlon, df_pop_str_edu, how="left", on=dep_com_labels),
        df_popfiscal, how="left")

    if fill_nn:
        drop_cols = [
            'Code du département', 'Libellé du département',
            'Code de la commune', 'Libellé de la commune', 'Abstentions',
            'Blancs et nuls', 'Choix A', 'Choix B', 'Choix A (%)',
            'Choix B (%)', 'Abstentions (%)', 'Blancs (%)',
            'Latitude', 'Longitude']
        fix_columns = np.setdiff1d(df.columns, drop_cols)
        for col_ in fix_columns:
            print("Filling missing values for ", col_)
            _fill_missing_nn(df, col_)
    return df, df_referendum, df_latlon, df_pop_str_edu, df_popfiscal


def _fill_missing_nn(df, colname):
    xall = df[["Latitude", "Longitude", "Population en 2013 (princ)"]].values
    yall = df[colname].values.copy()
    valid = np.logical_and(
        np.isfinite(yall),
        np.isfinite(xall[:, 2]))
    y = yall[valid]
    x = xall[valid, :]
    nninterp = NearestNDInterpolator(x, y, rescale=True)

    invalid = np.where(np.isnan(yall))[0]
    for ix_ in invalid:
        if np.all(np.isfinite(xall[ix_])):
            yall[ix_] = nninterp(xall[ix_])

    df[colname] = yall
    return df


#def convert_commune_to_department(df):
#    depcode = df["Code du département"].unique()
#    df_deplist = list(df.groupby("Code du département"))
#    df_departments = pd.DataFrame(columns=df.columns)
#    for ix_, df_dep_ in df_deplist:
#        pass


def read_data(fill_nn=True, force_read=False):
    """Load and preprocess the data"""
    pfile = "pickle/preprocessed_data.p"
    try:
        if not fill_nn or force_read:
            raise FileNotFoundError
        with io.open(pfile, 'rb') as f:
            pdata = pickle.load(f)
        df = pdata["df"]
        df_referendum = pdata["df_referendum"]
        df_latlon = pdata["df_latlon"]
        df_pop_str_edu = pdata["df_pop_str_edu"]
        df_popfiscal = pdata["df_popfiscal"]
    except FileNotFoundError:
        # Load original referendum data
        print("Loading reference data")
        df_referendum = read_referendum_data()
        # Get latitude and longitude
        print("Loading lat-lon data")
        df_latlon = read_latlon_data()
        # Read population structure data
        print("Loading population structure data")
        df_popstr = read_popstruct_data()
        # Read education data
        print("Loading education data")
        df_popedu = read_popeducation_data()
        # Read fiscal data
        print("Loading fiscal data")
        df_popfiscal = read_popfiscal_data()

        df, df_referendum, df_latlon, df_pop_str_edu, df_popfiscal = (
            preprocess(df_referendum, df_latlon, df_popstr, df_popedu,
                       df_popfiscal, fill_nn))
        # Drop rows which are outside mainland France
        df = df.iloc[:36565]

        if fill_nn:
            pdata = dict(
                df=df,
                df_referendum=df_referendum,
                df_latlon=df_latlon,
                df_pop_str_edu=df_pop_str_edu,
                df_popfiscal=df_popfiscal)
            with io.open(pfile, 'wb') as f:
                pdata = pickle.dump(pdata, f)

    return df, df_referendum, df_latlon, df_pop_str_edu, df_popfiscal


def compute_corr(df):
    cols = ["Choix A (%)", "Choix B (%)", "Abstentions (%)", "Blancs (%)"]
    dfcorr = df.corr().loc[cols].T
    # Drop the first 12 rows
    dfcorr = dfcorr.iloc[10:]
    return dfcorr


def plot_choice_scatter(df):
    m = Basemap(width=1500000, height=1500000,  projection='lcc',
                resolution='l',
                lat_0=47., lon_0=2.)
    x, y = m(df["Longitude"].values, df["Latitude"].values)
    # df["x"] = x
    # df["y"] = y
    for c in ["Choix A (%)", "Choix B (%)", "Abstentions (%)", "Blancs (%)"]:
        df.plot(kind="scatter", x="Longitude", y="Latitude",
                alpha=0.6, s=df["Inscrits"]/5000, c=c,
                cmap=plt.get_cmap("jet"), sharex=False)
        # m.drawcoastlines()
        # m.drawcountries()


df, df_referendum, df_latlon, df_pop_str_edu, df_popfiscal = (
    read_data(fill_nn=True))
dfcorr = compute_corr(df)
