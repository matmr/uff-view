"""
Created on 20. maj 2014

TODO: A lot of refactoring.


@author: Matjaz
"""
import time

import itertools

import pandas as pd

import numpy as np

import pyuff as uff

import lib.utils as ut

# import _transformations as tr

types = {}
types[15] = 'Geometry'
types[82] = 'Lines'
types[151] = 'Header'
types[2411] = 'Geometry'
types[164] = 'Units'
types[58] = 'Measurement'
types[55] = 'Analysis'
types[2420] = 'Coor. sys.'
types[18] = 'Coor. sys.'


class ModalData(object):
    """The data object holds all measurement, results and geometry data
    """

    def __init__(self):
        """
        Constructor
        """
        # Tables
        self.tables = dict()

        # Holds the tables, populated by importing a uff file.
        # TODO: This is temporary? Maybe, maybe not, might be
        # a good idea to have some reference of imported data!
        self.uff_import_tables = dict()

        self.create_info_table()
        self.create_geometry_table()
        self.create_measurement_table()
        self.create_analysis_table()
        self.create_lines_table()

        # Set model id
        self.model_id = 0

    def create_info_table(self):
        """Creates an empty info table."""
        self.tables['info'] = pd.DataFrame(columns=['model_id', 'uffid', 'value'])

    def create_geometry_table(self):
        """Creates an empty geometry table."""
        self.tables['geometry'] = pd.DataFrame(
            columns=['model_id', 'uffid', 'node_nums', 'x', 'y', 'z', 'thx', 'thy', 'thz', 'disp_cs', 'def_cs',
                     'color'])

    def create_measurement_table(self):
        """Creates an empty measurement table."""
        self.tables['measurement_index'] = pd.DataFrame(columns=['model_id', 'measurement_id', 'uffid', 'field_type',
                                                                 'func_type', 'rsp_node', 'rsp_dir', 'ref_node',
                                                                 'ref_dir', 'abscissa_spec_data_type',
                                                                 'ordinate_spec_data_type', 'orddenom_spec_data_type'])

        self.tables['measurement_values'] = pd.DataFrame(columns=['model_id', 'measurement_id', 'frq', 'amp'])
        self.tables['measurement_values'].amp = self.tables['measurement_values'].amp.astype('complex')

    def create_analysis_table(self):
        """Creates an empty analysis table."""
        self.tables['analysis_index'] = pd.DataFrame(
            columns=['model_id', 'analysis_id', 'uffid', 'field_type', 'analysis_type',
                     'data_ch', 'spec_data_type', 'load_case', 'mode_n',
                     'eig', 'freq', 'freq_step_n', 'node_nums', 'rsp_node', 'rsp_dir',
                     'ref_node', 'ref_dir', 'data_type'])
        self.tables['analysis_values'] = pd.DataFrame(columns=['model_id', 'analysis_id', 'r1', 'r2', 'r3'])

        self.tables['analysis_index'].eig = self.tables['analysis_index'].eig.astype('complex')
        self.tables['analysis_values'].r1 = self.tables['analysis_values'].r1.astype('complex')
        self.tables['analysis_values'].r2 = self.tables['analysis_values'].r2.astype('complex')
        self.tables['analysis_values'].r3 = self.tables['analysis_values'].r3.astype('complex')

    def create_lines_table(self):
        """Creates an empty lines table."""
        self.tables['lines'] = pd.DataFrame(['model_id', 'uffid', 'id', 'field_type', 'trace_num',
                                             'color', 'n_nodes', 'trace_id', 'pos', 'node'])

    def new_model(self, model_id, entries=dict()):
        """Set new model id. Values can be set through entries dictionary, for each
        value left unset, default will be used."""
        fields = {'db_app': 'ModalData', 'time_db_created': time.strftime("%d-%b-%y %H:%M:%S"),
                  'time_db_saved': time.strftime("%d-%b-%y %H:%M:%S"), 'program': 'modaldata.py',
                  'model_name': 'DefaultName', 'description': 'DefaultDecription', 'units_code': 9,
                  'temp': 1.0, 'temp_mode': 1, 'temp_offset': 1.0, 'length': 1.0, 'force': 1.0,
                  'units_description': 'User unit system'}

        for key in entries:
            fields[key] = entries[key]

        input = [[model_id, None, field, value] for field, value in fields.items()]

        new_model = pd.DataFrame(input, columns=['model_id', 'uffid', 'field', 'value'])

        self.tables['info'] = pd.concat([self.tables['info'], new_model], ignore_index=True)

    def import_uff(self, fname):
        """Pull data from uff."""
        # .. TODO: Keys from 100 on are now used for uff imports. This
        #       value should be automatically updated somehow. Also only
        #       one file import is predicted ...
        uffdata = ModalDataUff(fname)

        for key in self.tables.keys():
            if key in uffdata.tables:
                uffdata.tables[key].model_id += 100
                self.tables[key] = pd.concat([self.tables[key], uffdata.tables[key]], ignore_index=True)
                self.uff_import_tables[key] = ''

        self.file_structure = uffdata.file_structure

    def export_to_uff(self, fname):
        """Export data to uff."""
        uffwrite = uff.UFF(fname)

        model_ids = self.tables['info'].model_id.unique()

        if len(model_ids) == 0:
            print('Warning: Empty tables. (No model_ids found).')
            return False

        for model_id in model_ids:

            # -- Write info.
            dfi = self.tables['info']
            dfi = dfi[dfi.model_id == model_id]

            if len(dfi) != 0:

                dset_info = {'db_app': dfi[dfi.field == 'db_app'],
                             'model_name': dfi[dfi.field == 'model_name'],
                             'description': dfi[dfi.field == 'description'],
                             'program': dfi[dfi.field == 'program']}
                dset_units = {'units_code': dfi[dfi.field == 'units_code'],
                              'units_description': dfi[dfi.field == 'units_description'],
                              'temp_mode': dfi[dfi.field == 'temp_mode'],
                              'length': dfi[dfi.field == 'length'],
                              'force': dfi[dfi.field == 'force'],
                              'temp': dfi[dfi.field == 'temp'],
                              'temp_offset': dfi[dfi.field == 'temp_offset']}

                for key in dset_info.keys():
                    dset_info[key] = dset_info[key].value.values[0]
                dset_info['type'] = 151

                for key in dset_units.keys():
                    dset_units[key] = dset_units[key].value.values[0]
                dset_units['type'] = 164

                uffwrite._write_set(dset_info, mode='add')
                uffwrite._write_set(dset_units, mode='add')

            # -- Write Geometry.
            dfg = self.tables['geometry']
            dfg = dfg[dfg.model_id == model_id]

            if len(dfg) != 0:

                # .. First the coordinate systems. Mind the order of angles (ZYX)
                size = len(dfg)
                local_cs = np.zeros((size * 4, 3), dtype=float)
                th_angles = dfg[['thz', 'thy', 'thx']].values

                for i in range(size):
                    local_cs[i * 4:i * 4 + 3, :] = ut.zyx_euler_to_rotation_matrix(th_angles[i, :])
                    local_cs[i * 4 + 3, :] = 0.0

                dset_cs = {'local_cs': local_cs, 'nodes': dfg[['node_nums']].values, 'type': 2420}
                uffwrite._write_set(dset_cs, mode='add')

                # .. Then points.
                dset_geometry = {'grid_global': dfg[['node_nums', 'x', 'y', 'z']].values,
                                 'export_cs_number': 0,
                                 'cs_color': 8,
                                 'type': 2411}

                uffwrite._write_set(dset_geometry, mode='add')

            # -- Write Measurements.
            dfi = self.tables['measurement_index']
            dfi = dfi[dfi.model_id == model_id]

            if len(dfi) != 0:

                dfv = self.tables['measurement_values']
                dfv = dfv[dfv.model_id == model_id]

                for id, measurement in dfi.iterrows():
                    data = dfv[dfv.measurement_id == measurement.measurement_id]

                    dsets = {'type': measurement['field_type'],
                             'func_type': measurement['func_type'],
                             'data': data['amp'].values.astype('complex'),
                             'x': data['frq'].values,
                             'rsp_node': measurement['rsp_node'],
                             'rsp_dir': measurement['rsp_dir'],
                             'ref_node': measurement['ref_node'],
                             'ref_dir': measurement['ref_dir'],
                             'rsp_ent_name': 'NONE', 'ref_ent_name': 'NONE'}

                    if pd.isnull(measurement['abscissa_spec_data_type']):
                        dsets['abscissa_spec_data_type'] = 0
                    else:
                        dsets['abscissa_spec_data_type'] = measurement['abscissa_spec_data_type']

                    if pd.isnull(measurement['ordinate_spec_data_type']):
                        dsets['ordinate_spec_data_type'] = 0
                    else:
                        dsets['ordinate_spec_data_type'] = measurement['ordinate_spec_data_type']

                    if pd.isnull(measurement['orddenom_spec_data_type']):
                        dsets['orddenom_spec_data_type'] = 0
                    else:
                        dsets['orddenom_spec_data_type'] = measurement['orddenom_spec_data_type']

                    uffwrite._write_set(dsets, mode='add')


class ModalDataUff(object):
    '''
    Reads the uff file and populates the following pandas tables:
    -- ModalData.measurement_index : index of all measurements from field 58
    -- ModalData.geometry          : index of all points with CS from fields 2411 and 15
    -- ModalData.info              : info about measurements

    Based on the position of field in the uff file, uffid is assigned to each field in the following
    maner: first field, uffid = 0, second field, uffid = 1 and so on. Columns are named based on keys
    from the UFF class if possible. Fields uffid and field_type (type of field, eg. 58) are added.

    Geometry table combines nodes and their respective CSs, column names are altered.
    '''

    def __init__(self, fname='../../unvread/data/shield.uff', maxkey=100):
        '''
        Constructor

        '''
        self.uff_object = uff.UFF(fname)

        self.uff_types = self.uff_object.get_set_types()
        # print(self.uff_types)

        # Models
        self.models = dict()

        # Tables
        self.tables = dict()

        # Coordinate-system tables
        self.localcs = pd.DataFrame(columns=['model_id', 'uffidcs', 'node_nums', 'x1', 'x2', 'x3',
                                             'y1', 'y2', 'y3',
                                             'z1', 'z2', 'z3'])

        self.localeul = pd.DataFrame(columns=['model_id', 'uffidcs', 'node_nums', 'thx', 'thy', 'thz'])

        # File structure.
        self.file_structure = ['%5d %-10s' % (field, types[field]) for field in self.uff_types]

        self.create_model()

    def create_model(self):
        """Scans the uff file and creates a model from
        geometries and data, which is then populated. The models
        are grouped based on the field 151!"""
        # -- Scan geometries, each geometry is one model.
        mnums = list(np.nonzero(self.uff_types == 151)[0])

        if len(mnums) == 0:
            mnums = list(np.nonzero(self.uff_types == 164)[0])
            # -- What if there is no geometry? Only one model then I guess ...

        if len(mnums) == 0:
            print('Warning: There is no INFO or UNITS field!')
            self.models[0] = range(len(self.uff_types))
            # .. TODO: You have to pass this warning on.
        else:
            # .. Define intervals, by sequential order, for each model.
            for model_id, num in enumerate(mnums):
                if model_id == (len(mnums) - 1):
                    self.models[model_id] = range(num, len(self.uff_types))
                else:
                    # .. Last model has special treatment ([x:] instead of [x:y])
                    self.models[model_id] = range(num, mnums[model_id + 1])

        for model_id, model in self.models.items():
            self.populate_model(model_id, model)

            # print(self.models)
            # print(self.uff_types)

    def populate_model(self, model_id, model):
        """Read all data for each model."""
        model = list(model)

        self.gen_measurement_table(model_id, model)
        self.gen_geometry_table(model_id, model)
        self.gen_analysis_table(model_id, model)
        self.gen_lines_table(model_id, model)
        self.gen_info_table(model_id, model)

        # .. TODO: Here is the place to check for connections between
        #       fields, other than by sequential order. Check if LMS
        #       writes anything. (It does not!)

    def gen_measurement_table(self, model_id, model):
        """Read measurements."""
        mnums = np.nonzero(self.uff_types[model] == 58)[0]
        mnums += model[0]

        if len(mnums) == 0:
            return False

        mlist = []
        dlist = pd.DataFrame()

        # .. Create field list.
        sdata = self.uff_object.read_sets(mnums[0])
        fields = ['model_id', 'measurement_id', 'uffid', 'field_type']
        fields.extend([key for key in sdata.keys() if not ('x' in key or 'data' in key)])

        for mnum in list(mnums):
            dlist_ = pd.DataFrame()

            sdata = self.uff_object.read_sets(mnum)

            # .. Setup a new line in measurement index table.
            line = [model_id, mnum, mnum, 58]
            line.extend([sdata[key] for key in fields if
                         not ('uffid' in key or 'field_type' in key or 'model_id' in key or 'measurement_id' in key)])
            mlist.append(line)

            # TODO: Uredi podporo za kompleksne vrednosti tukaj. NE štima še čist!
            dlist_['frq'] = sdata['x']
            dlist_['amp'] = sdata['data']
            dlist_['amp'] = dlist_['amp'].astype('complex')
            dlist_['amp'] = sdata['data']
            dlist_['uffid'] = mnum
            dlist_['measurement_id'] = mnum
            dlist_['model_id'] = model_id

            dlist = pd.concat([dlist, dlist_], ignore_index=True)

        if 'measurement_index' in self.tables:
            self.tables['measurement_index'] = pd.concat(
                [self.tables['measurement_index'], pd.DataFrame(mlist, columns=fields)], ignore_index=True)
            self.tables['measurement_values'] = pd.concat([self.tables['measurement_values'], dlist], ignore_index=True)
        else:
            self.tables['measurement_index'] = pd.DataFrame(mlist, columns=fields)
            self.tables['measurement_values'] = dlist

        return True

    def getlocalcs(self, model_id, model):
        '''Read cs fields and convert all to euler angels.'''

        mnums = np.nonzero(self.uff_types[model] == 2420)[0]
        # mnums.extend(list(np.nonzero(self.uff_types==18)[0]))
        mnums.sort()

        mnums += model[0]

        if len(mnums) == 0:
            return False

        # So ... the cs come in different flavours. One is simply the three
        #   vectors defining the axes.
        # self.localcs = pd.DataFrame(columns=['model_id', 'uffidcs', 'node_nums', 'x1', 'x2', 'x3',
        #                                      'y1', 'y2', 'y3',
        #                                      'z1', 'z2', 'z3'])

        # ... another one is the three euler angles, using axes z, y' and x''. This is
        #       what LMS test lab uses internally. Perhaps there is a uff field with
        #       this data but i havent searched yet.
        # self.localeul = pd.DataFrame(columns=['model_id', 'uffidcs', 'node_nums', 'thx', 'thy', 'thz'])

        # ... what is left is a cumbersome definition (data set #18) with the point of origin,
        #       a point on the x axis and then another point on the xz plane :S
        # Am thinking about just converting this one directly, no sense in creating another pandas
        #   table -- who will use this definition?


        mlist = []
        for mnum in mnums:
            sdata = self.uff_object.read_sets(mnum)

            leu = pd.DataFrame(columns=['model_id', 'uffidcs', 'node_nums', 'thx', 'thy', 'thz'])
            lcs = pd.DataFrame(columns=['model_id', 'uffidcs', 'node_nums', 'x1', 'x2', 'x3',
                                        'y1', 'y2', 'y3',
                                        'z1', 'z2', 'z3'])

            # # So the wors case scenario is we have a #18 field. It may be
            # #   a good idea to first calculate local coordinate axes.
            # if sdata['type'] == 18:
            #     # x-axis
            #     x = np.linalg.norm(sdata['x_point']-sdata['ref_o'])
            #
            #     # y-axis



            lcs['node_nums'] = sdata['CS_sys_labels']
            leu['node_nums'] = sdata['CS_sys_labels']

            # .. First calculate euler angles.
            #     (see http://nghiaho.com/?page_id=846)
            thx = []
            thy = []
            thz = []
            for r in sdata['CS_matrices']:
                thx.append(np.arctan2(r[2, 1], r[2, 2]))
                thy.append(np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2)))
                thz.append(np.arctan2(r[1, 0], r[0, 0]))

            leu['thx'] = thx
            leu['thy'] = thy
            leu['thz'] = thz

            # .. Also save local cs.
            arr = np.array(sdata['CS_matrices']).ravel()
            lcs['x1'] = arr[::9]
            lcs['x2'] = arr[1::9]
            lcs['x3'] = arr[2::9]
            lcs['y1'] = arr[3::9]
            lcs['y2'] = arr[4::9]
            lcs['y3'] = arr[5::9]
            lcs['z1'] = arr[6::9]
            lcs['z2'] = arr[7::9]
            lcs['z3'] = arr[8::9]

            lcs['uffidcs'] = mnum
            leu['uffidcs'] = mnum

            lcs['model_id'] = model_id
            leu['model_id'] = model_id

            self.localcs = self.localcs.append(lcs)
            self.localeul = self.localeul.append(leu)

            return True

    def gen_geometry_table(self, model_id, model):
        '''Read geometry.'''
        mnums = list(np.nonzero(self.uff_types[model] == 2411)[0])
        mnums.extend(list(np.nonzero(self.uff_types[model] == 15)[0]))
        mnums.sort()

        mnums = np.array(mnums)
        mnums += model[0]

        if len(mnums) == 0:
            return False

        mlist = []
        dlist = pd.DataFrame(columns=['model_id', 'uffid', 'node_nums', 'x', 'y', 'z', 'disp_cs', 'def_cs', 'color'])

        for mnum in list(mnums):
            sdata = self.uff_object.read_sets(mnum)

            # TODO: line below is out of place?
            mlist.append([mnum, sdata['type']])

            dlistt = pd.DataFrame()

            dlistt['x'] = sdata['x']
            dlistt['y'] = sdata['y']
            dlistt['z'] = sdata['z']
            dlistt['node_nums'] = sdata['node_nums']
            dlistt['disp_cs'] = sdata['disp_cs']
            dlistt['def_cs'] = sdata['def_cs']
            dlistt['color'] = sdata['color']
            dlistt['uffid'] = mnum
            dlistt['model_id'] = model_id

            dlist = dlist.append(dlistt, ignore_index=True)

        dlist['node_nums'] = dlist['node_nums'].astype(int)
        dlist['color'] = dlist['color'].astype(int)
        dlist['def_cs'] = dlist['def_cs'].astype(int)
        dlist['disp_cs'] = dlist['disp_cs'].astype(int)

        # TODO: I dont think i like the way the getlocalcs is initiated and then used.
        cspresent = self.getlocalcs(model_id, model)

        # TODO: Leave this for the end, when all models are scaned? Or maybe not!!
        if cspresent:
            dlist = pd.merge(dlist, self.localeul, on=['node_nums', 'model_id'])[
                ['model_id', 'uffid', 'node_nums', 'x', 'y', 'z',
                 'thx', 'thy', 'thz']].sort_values(by='uffid')
        # self.geometry = pd.merge(self.geometry, self.localeul, on='node_nums')[['uffid', 'nodenums', 'x', 'y', 'z',
        #                                                                            'thx', 'thy', 'thz']].sort(['mnum'])
        else:
            dlist = dlist[['model_id', 'uffid', 'node_nums', 'x', 'y', 'z']]
            dlist['thx'] = None
            dlist['thy'] = None
            dlist['thz'] = None

        if 'geometry' in self.tables:
            self.tables['geometry'] = pd.concat([self.tables['geometry'], dlist], ignore_index=True)
        else:
            self.tables['geometry'] = dlist

        return True

    def gen_info_table(self, model_id, model):
        """Read info."""
        mnums = list(np.nonzero(self.uff_types[model] == 151)[0])
        mnums.extend(list(np.nonzero(self.uff_types[model] == 164)[0]))
        mnums.sort()

        mnums = np.array(mnums)
        mnums += model[0]

        if len(mnums) == 0:
            # self.info = None
            return False

        mlist = []

        for mnum in mnums:
            sdata = self.uff_object.read_sets(mnum)
            for key, val in sdata.items():
                mlist.append([model_id, mnum, key, val])

        if 'info' in self.tables:
            self.tables['info'] = pd.concat(
                [self.tables['info'], pd.DataFrame(mlist, columns=['model_id', 'uffid', 'field', 'value'])],
                ignore_index=True)
        else:
            self.tables['info'] = pd.DataFrame(mlist, columns=['model_id', 'uffid', 'field', 'value'])

        return True

    def gen_analysis_table(self, model_id, model):
        '''Read analysis data.'''
        mnums = np.nonzero(self.uff_types[model] == 55)[0]

        mnums += model[0]

        if len(mnums) == 0:
            # self.info = None
            return False

        mlist = []

        # Columns.
        sdata = self.uff_object.read_sets(mnums[0])
        cols = ['model_id', 'uffid', 'field_type']
        cols.extend([key for key in sdata if not ('r1' in key or 'r2' in key or 'r3' in key or 'node_nums' in key)])

        # Table for holding arrays of data.
        analysis_values = pd.DataFrame(columns=['model_id', 'uffid', 'node_nums', 'r1', 'r2', 'r3'])

        for mnum in mnums:
            sdata = self.uff_object.read_sets(mnum)

            # Index values.
            line = [model_id, mnum, 55]
            line.extend(
                [sdata[key] for key in cols if not ('model_id' in key or 'uffid' in key or 'field_type' in key)])
            mlist.append(line)

            # Array values.
            tmp_df = pd.DataFrame()
            tmp_df['node_nums'] = sdata['node_nums']
            tmp_df['r1'] = sdata['r1']
            tmp_df['r2'] = sdata['r2']
            tmp_df['r3'] = sdata['r3']
            tmp_df['model_id'] = model_id
            tmp_df['uffid'] = mnum

            analysis_values = pd.concat([analysis_values, tmp_df], ignore_index=True)

        if 'analysis_values' in self.tables:
            self.tables['analysis_values'] = pd.concat([self.tables['analysis_values'], analysis_values],
                                                       ignore_index=True)
            self.tables['analysis_index'] = pd.concat(
                [self.tables['analysis_index'], pd.DataFrame(mlist, columns=cols)])
        else:
            self.tables['analysis_values'] = analysis_values
            self.tables['analysis_index'] = pd.DataFrame(mlist, columns=cols)

        return True

    def gen_lines_table(self, model_id, model):
        """Read line data."""

        # .. Splits list on a value (0).
        def isplit(iterable, spliters):
            return [list(g) for k, g in itertools.groupby(iterable, lambda x: x in spliters) if not k]

        mnums = np.nonzero(self.uff_types[model] == 82)[0]
        mnums += model[0]

        if len(mnums) == 0:
            # self.lines = None
            return False

        # .. Each line is for one node, trace_id connects nodes for one element, pos indicates
        #    the order of nodes.
        cols = ['model_id', 'uffid', 'id', 'field_type', 'trace_num', 'color', 'n_nodes', 'trace_id', 'pos', 'node']
        lines = pd.DataFrame(columns=cols)
        trace_id = 0

        for mnum in mnums:
            sdata = self.uff_object.read_sets(mnum)

            elements = isplit(list(sdata['lines']), [0.0])

            for element in elements:
                tmp_df = pd.DataFrame(columns=cols)
                tmp_df['node'] = element
                tmp_df['pos'] = range(len(element))
                tmp_df['trace_id'] = trace_id
                trace_id += 1

                tmp_df['uffid'] = mnum
                tmp_df['model_id'] = model_id
                tmp_df['id'] = sdata['id']
                tmp_df['field_type'] = 82
                tmp_df['trace_num'] = sdata['trace_num']
                tmp_df['color'] = sdata['color']
                tmp_df['n_nodes'] = len(element)

                lines = pd.concat([lines, tmp_df], ignore_index=True)

        if 'lines' in self.tables:
            self.tables['lines'] = pd.concat([self.tables['lines'], lines], ignore_index=True)
        else:
            self.tables['lines'] = lines

        return True


if __name__ == '__main__':
    # Initialize the object - creates empty tables that can be filled.
    obj = ModalData()

    # Create new model and remember the ID. Additionally, a dictionary can be
    # provided with info values -- see docstring for ModalData.new_model().
    obj.new_model(7)

    # Fill/change tables. By hand for now.
    # ...

    # Here a uff file is imported to get some data.
    obj.import_uff(r'sampledata\shield_short.unv')

    # Export to uff. Data is appended, so make sure that the file does not exist.
    obj.export_to_uff('test_www.unv')