import os
from itertools import repeat
import numpy as np
import pandas as pd

SIMPLE_FEATURE_COLUMNS = ['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]', 'avg_cs[0]',
       'avg_cs[1]', 'avg_cs[2]', 'avg_cs[3]', 'ndof', 'MatchedHit_TYPE[0]',
       'MatchedHit_TYPE[1]', 'MatchedHit_TYPE[2]', 'MatchedHit_TYPE[3]',
       'MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]',
       'MatchedHit_X[3]', 'MatchedHit_Y[0]', 'MatchedHit_Y[1]',
       'MatchedHit_Y[2]', 'MatchedHit_Y[3]', 'MatchedHit_Z[0]',
       'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]',
       'MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]',
       'MatchedHit_DX[3]', 'MatchedHit_DY[0]', 'MatchedHit_DY[1]',
       'MatchedHit_DY[2]', 'MatchedHit_DY[3]', 'MatchedHit_DZ[0]',
       'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]',
       'MatchedHit_T[0]', 'MatchedHit_T[1]', 'MatchedHit_T[2]',
       'MatchedHit_T[3]', 'MatchedHit_DT[0]', 'MatchedHit_DT[1]',
       'MatchedHit_DT[2]', 'MatchedHit_DT[3]', 'Lextra_X[0]', 'Lextra_X[1]',
       'Lextra_X[2]', 'Lextra_X[3]', 'Lextra_Y[0]', 'Lextra_Y[1]',
       'Lextra_Y[2]', 'Lextra_Y[3]', 'NShared', 'Mextra_DX2[0]',
       'Mextra_DX2[1]', 'Mextra_DX2[2]', 'Mextra_DX2[3]', 'Mextra_DY2[0]',
       'Mextra_DY2[1]', 'Mextra_DY2[2]', 'Mextra_DY2[3]', 'FOI_hits_N', 'PT', 'P']

TRAIN_COLUMNS = ["label", "weight"]

FOI_COLUMNS = ["FOI_hits_X", "FOI_hits_Y", "FOI_hits_T", "FOI_hits_Z",
               "FOI_hits_DX", "FOI_hits_DY", "FOI_hits_DZ", "FOI_hits_S", "FOI_hits_DT"]

ID_COLUMN = "id"

# Given 4 staions in problem itself
N_STATIONS = 4
FEATURES_PER_STATION = 8
N_FOI_FEATURES = N_STATIONS*FEATURES_PER_STATION
# The value to use for stations with missing hits
# when computing FOI features
EMPTY_FILLER = 1000

# Examples on working with the provided files in different ways

VERSION = "v2"
# hdf is all fine - but it requires unpickling the numpy arrays
# which is not guranteed
def load_train_hdf(path):
    return pd.concat([
        pd.read_hdf(os.path.join(path, "train_part_%i_%s.hdf" % (i, VERSION)))
        for i in (1, 2)], axis=0, ignore_index=True)


def load_data_csv():
    train = pd.concat([
        pd.read_csv('random_part1/output1.csv',
                    usecols= [ID_COLUMN] + SIMPLE_FEATURE_COLUMNS + FOI_COLUMNS + TRAIN_COLUMNS,
                    index_col=ID_COLUMN)
        for i in (1, 2)], axis=0, ignore_index=True)
    test = pd.read_csv('test_public_v2.csv',
                       usecols=[ID_COLUMN] + SIMPLE_FEATURE_COLUMNS + FOI_COLUMNS, index_col=ID_COLUMN)
    return train, test


def parse_array(line, dtype=np.float32):
    return np.fromstring(line[1:-1], sep=" ", dtype=dtype)


def load_full_test_csv(path):
    converters = dict(zip(FOI_COLUMNS, repeat(parse_array)))
    types = dict(zip(SIMPLE_FEATURE_COLUMNS, repeat(np.float32)))
    test = pd.read_csv(os.path.join(path, "test_public_%s.csv.gz" % VERSION),
                       index_col="id", converters=converters,
                       dtype=types,
                       usecols=[ID_COLUMN]+SIMPLE_FEATURE_COLUMNS+FOI_COLUMNS)
    return test


def find_closest_hit_per_station(row):
    result = np.empty(N_FOI_FEATURES, dtype=np.float32)
    closest_x_per_station = result[0:4]
    closest_y_per_station = result[4:8]
    closest_T_per_station = result[8:12]
    closest_z_per_station = result[12:16]
    closest_dx_per_station = result[16:20]
    closest_dy_per_station = result[20:24]
    closest_dz_per_station = result[24:28]
    closest_dt_per_station = result[28:32]
    
    for station in range(4):
        count = 0
        new_row = row['FOI_hits_S'][1: -1].split(" ")
        row_x = row['FOI_hits_X'][1: -1].split(" ")
        row_y = row['FOI_hits_Y'][1: -1].split(" ")
        row_z = row['FOI_hits_Z'][1: -1].split(" ")
        row_t = row['FOI_hits_T'][1: -1].split(" ")
        row_dx = row['FOI_hits_DX'][1: -1].split(" ")
        row_dy = row['FOI_hits_DY'][1: -1].split(" ")
        row_dz = row['FOI_hits_DZ'][1: -1].split(" ")
        row_dt = row['FOI_hits_DT'][1: -1].split(" ")
        row_x = list(filter(None, row_x))
        row_y = list(filter(None, row_y))
        row_z = list(filter(None, row_z))
        row_t = list(filter(None, row_t))
        row_dx = list(filter(None, row_dx))
        row_dy = list(filter(None, row_dy))
        row_dz = list(filter(None, row_dz))
        row_dt = list(filter(None, row_dt))
        hit_index = []
        flag_count = 0
        for hit in new_row:
            flag_count = flag_count + 1
            hits = (int(hit) == station)
            if hits:
                count = count + 1
                hit_index.append(flag_count-1)
        if count==0:
            closest_x_per_station[station] = EMPTY_FILLER
            closest_y_per_station[station] = EMPTY_FILLER
            closest_T_per_station[station] = EMPTY_FILLER
            closest_z_per_station[station] = EMPTY_FILLER
            closest_dx_per_station[station] = EMPTY_FILLER
            closest_dy_per_station[station] = EMPTY_FILLER
            closest_dz_per_station[station] = EMPTY_FILLER
            closest_dt_per_station[station] = EMPTY_FILLER
        else:
            x_distances_2 = []
            y_distances_2 = []
            for numi in hit_index:
                x2 = (float(row["Lextra_X[%i]" % station]) - float(row_x[int(numi)]))**2
                x_distances_2.append(x2)
            x_distances_2 = np.array(x_distances_2)
            for numj in hit_index:
                y2 = (float(row["Lextra_Y[%i]" % station]) - float(row_y[int(numj)]))**2
                y_distances_2.append(y2)
            y_distances_2 = np.array(y_distances_2)
            distances_2 = (x_distances_2 + y_distances_2)
            closest_hit = np.argmin(distances_2)
            closest_x_per_station[station] = x_distances_2[closest_hit]
            closest_y_per_station[station] = y_distances_2[closest_hit]
            closest_T_per_station[station] = row_t[closest_hit]
            closest_z_per_station[station] = row_z[closest_hit]
            closest_dx_per_station[station] = row_dx[closest_hit]
            closest_dy_per_station[station] = row_dy[closest_hit]
            closest_dz_per_station[station] = row_dz[closest_hit]
            closest_dt_per_station[station] = row_dt[closest_hit]
    return result
