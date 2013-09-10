import csv
import numpy as np
import pandas as pd
from collections import namedtuple
import os

WORKPATH = os.path.join(os.environ['HOME'], 'Dropbox', 'Imperial',
                        'msc-project', 'BundleAdjustment', 'data')
INTEL = os.path.join(WORKPATH, '2d', 'intel', 'intel.g2o')
MANHATTAN = os.path.join(WORKPATH, '2d', 'manhattan3500', 'manhattanOlson3500.g2o')
NEW_COLLEGE_G2O = os.path.join(WORKPATH, 'ba', 'new-college', 'newcollege3500.g2o')

EDGE_COLS = ['from_index', 'to_index', 'x_coord', 'y_coord', 'theta_coord', 'omega_0_0',
             'omega_0_1', 'omega_0_2', 'omega_1_1', 'omega_1_2', 'omega_2_2']
VERTEX_COLS = ['index', 'x', 'y', 'theta']

Vertex = namedtuple('Vertex', VERTEX_COLS)
Edge = namedtuple('Edge', EDGE_COLS)


def drop_empty(row):
    return [item for item in row if item != '']


def quickLoad(filepath, delim=' ', skip=0):
    max_num_cols = 30
    df = pd.read_csv(filepath,
                     delimiter=delim,
                     skiprows=skip,
                     names=[str(x) for x in range(max_num_cols)]
                     ).dropna(axis=1, how='all')

    edges_mask = (~pd.isnull(df)).all(axis=1)
    
    edges = df.ix[edges_mask]
    vertices = df.ix[~edges_mask].dropna(axis=1)

    vertices.rename(columns={'0': 'label', '1': 'index'}, inplace=True)
    vertices.rename(columns={str(i): 'dim%d' % (i - 1) for
                             i in range(2, len(vertices.columns))},
                    inplace=True)
    # replace the VERTEX_XXX label to just be XXX (more informative)
    vertices['label'] = vertices['label'].apply(lambda val: val.split('_')[1])
    # reassign the vertices DataFrame
    edges.rename(columns={'0': 'label', '1': 'from_v', '2': 'to_v'}, inplace=True)
    edges.rename(columns={str(i): 'meas%d' % (i - 2) for
                          i in range(3, len(edges.columns))}, inplace=True)
    edges['label'] = edges['label'].apply(lambda label: label.split('_')[1])
    edges[['from_v', 'to_v']] = edges[['from_v', 'to_v']].astype(np.uint32)
    return vertices, edges


def loadFromFile(filepath):
    vertices = []
    edges = []
    with open(filepath, 'r') as fd:
        reader = csv.reader(fd, delimiter=' ')
        for row in reader:
            if 'VERTEX' in row[0].upper():
                vertices.append(np.float64(drop_empty(row[1:])))
            elif 'EDGE' in row[0].upper():
                edges.append(np.float64(drop_empty(row[1:])))

    edges_dataframe = pd.DataFrame(edges, columns=EDGE_COLS)
    vertices_dataframe = pd.DataFrame(vertices, columns=VERTEX_COLS)

    return edges_dataframe, vertices_dataframe


def g2o_to_iSAM(g2o_filepath):
    v, e = quickLoad(g2o_filepath)
    e.label = e.label.apply(lambda val: 'EDGE%d' % int(val[-1]))
    outpath = g2o_filepath.split('.')[0] + '.isam'
    e.sort(columns='to_v', ascending=True).to_csv(outpath, sep=' ',
                                                  header=False, index=False)
    return outpath
    
if __name__ == '__main__':
    print g2o_to_iSAM(INTEL)
    print g2o_to_iSAM(MANHATTAN)
