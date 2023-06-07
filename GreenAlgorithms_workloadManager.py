
## ~~~ TO BE EDITED TO BE TAILORED TO THE WORKLOAD MANAGER ~~~
##
## This script is designed for SLURM
##

import subprocess
import pandas as pd
from io import BytesIO
import datetime
import os
import numpy as np

class Helpers_WM():

    def convert_to_GB(self, memory, unit):
        '''
        Convert data quantity into GB.
        :param memory: [float] quantity to convert
        :param unit: [str] unit of `memory`, has to be in ['M', 'G', 'K']
        :return: [float] memory in GB.
        '''
        assert unit in ['M', 'G', 'K']
        if unit == 'M':
            memory /= 1e3
        elif unit == 'K':
            memory /= 1e6
        return memory

    def calc_ReqMem(self, x):
        '''
        Calculate the total memory required when submitting the job.
        :param x: [pd.Series] one row of sacct output.
        :return: [float] total required memory, in GB.
        '''
        mem_raw, n_nodes, n_cores = x['ReqMem'], x['NNodes'], x['NCPUS']

        # Bercik Modification: add this for cases when mem_raw is NaN
        #if pd.isnull(mem_raw): mem_raw = x['MaxRSS']
        #if pd.isnull(mem_raw): mem_raw = '0K'
        
        # Bercik Modification, appears sacct has different format on Niagara
        if self.cluster_info['cluster_name'] == 'Niagara':
            unit = str(mem_raw)[-1]
            per_coreOrNode = 'n'
        else:
            unit = str(mem_raw)[-2]
            per_coreOrNode = mem_raw[-1] 

        if str(unit) not in ['M', 'G', 'K']: # Bercik Modification: added
            # first debug attempt - look for memory in AllocTRES
            TRESstr = str(x.AllocTRES).split(',')
            TRESmem = [i for i in TRESstr if 'mem=' in i]
            if len(TRESmem)==0:
                resolved = False
            elif len(TRESmem)>1:
                resolved = False
            else:
                idx = TRESmem[0].find('=')
                if idx == -1:
                    resolved = False
                else:
                    mem_raw = TRESmem[0][idx+1:]
                    resolved = True
            if resolved:
                if self.cluster_info['cluster_name'] == 'Niagara':
                    unit = str(mem_raw)[-1]
                    per_coreOrNode = 'n'
                else:
                    unit = str(mem_raw)[-2]
                    per_coreOrNode = mem_raw[-1] 
            else:
                assert 'default_unit_RSS' in self.cluster_info, "Some values of MaxRSS don't have a unit. Please specify a default_unit_RSS in cluster_info.yaml"
                if self.args.verbose:
                    print("WARNING: Something wrong with calculating the memory unit for a run on {0}. Using default, '{1}'. Raw output:".format(x['Submit'],self.cluster_info['default_unit_RSS']))
                    print(x) 
                    print('From this, we were able to understand mem_raw = {0} and unit = {1}.'.format(mem_raw, unit))
                else:
                    print("WARNING: Something wrong with calculating the memory unit for a run on {0}. Using default, '{1}'. Use flag --verbose for more debugging info.".format(x['Submit'],self.cluster_info['default_unit_RSS']))
                unit = self.cluster_info['default_unit_RSS']
        
        # Bercik Modification, appears Niagara is always mem_raw = 175000M*NNodes
        if self.cluster_info['cluster_name'] == 'Niagara':
            try:
                memory = float(mem_raw[:-1])
            except:
                if self.args.verbose:
                    print('WARNING: Something wrong with the memory for a run on {0}. Using default 175000M*NNodes. Raw line output:'.format(x['Submit']))
                    print(x) 
                    print('From this, we were able to understand mem_raw = {0}, unit = {1}, NNodes={2}.'.format(mem_raw, unit, n_nodes))
                else:
                    print('WARNING: Something wrong with the memory for a run on {0}. Using default 175000M*NNodes. Use flag --verbose for more debugging info.'.format(x['Submit']))
                memory = 175000
                unit = 'M'
        else:
            try:
                memory = float(mem_raw[:-2])
            except:
                # someone else will have to figure out what is going on below for other clusters. Take inspiration from my debugging above.
                print('ERROR: Something wrong with the memory. Raw line output:')
                print(x) 
                print('This is what it understood:')
                print('mem_raw = ', mem_raw)
                print('unit = ', unit)
                print('n_nodes = ', n_nodes)
                print('n_cores = ', n_cores)
                exit()

        # Convert memory to GB
        memory = self.convert_to_GB(memory,unit)

        # Multiply by number of nodes/cores
        if str(per_coreOrNode) not in ['n','c']:
            print('WARNING: Something wrong with per_coreOrNode. Using default per_coreOrNode=n. Raw line output:')
            print(x) 
            per_coreOrNode = 'n'
        if per_coreOrNode == 'c':
            memory *= n_cores
        else:
            memory *= n_nodes

        return memory

    def clean_RSS(self, x):
        '''
        Clean the RSS value in sacct output.
        :param x: [NaN or str] the RSS value, either NaN or of the form '2745K'
        (optionally, just a number, we then use default_unit_RSS from cluster_info.yaml as unit).
        :return: [float] RSS value, in GB.
        '''
        if pd.isnull(x.MaxRSS):
            # NB if no info on MaxRSS, we assume all memory was used
            memory = -1
        elif x.MaxRSS=='0':
            memory = 0
        else:
            assert type(x.MaxRSS) == str
            # Special case for the situation where MaxRSS is of the form '154264' without a unit.
            if x.MaxRSS[-1].isalpha():
                memory = self.convert_to_GB(float(x.MaxRSS[:-1]),x.MaxRSS[-1])
            else:
                assert 'default_unit_RSS' in self.cluster_info, "Some values of MaxRSS don't have a unit. Please specify a default_unit_RSS in cluster_info.yaml"
                memory = self.convert_to_GB(float(x.MaxRSS), self.cluster_info['default_unit_RSS'])

        return memory

    def clean_UsedMem(self, x):
        if x.UsedMem_ == -1:
            # NB when MaxRSS didn't store any values, we assume that "memory used = memory requested"
            return x.ReqMemX
        else:
            return x.UsedMem_

    def clean_partition(self, x):
        '''
        Clean the partition field, by replacing NaNs with empty string
        and selecting just one partition per job.
        :param x: [str] partition or comma-seperated list of partitions
        :param cluster_info: [dict]
        :return: [str] one partition or empty string
        '''
        if pd.isnull(x.Partition):
            return ''
        else:
            L_partitions = x.Partition.split(',')
            if x.WallclockTimeX.total_seconds() > 0:
                # Multiple partitions logged is only an issue for jobs that never started,
                # for the others, only the used partition is logged
                if len(L_partitions) > 1:
                    print(f"\n-!- WARNING: Multiple partitions logged on a job than ran: {x.JobID} - {x.Partition} (using the first one)\n")
            return L_partitions[0]
        
    def clean_hyperthreading(self, x):
        '''
        Clean the NCPUS field by accounting for hyperthreading.
        On Niagara, NCPUS will often be double of the actual number of CPUs used,
        and we must be careful of this to not double-count resources used.
        Uses the fact that 'billing=n' in AllocTRES contains the correct number of CPUs.
        '''
        TRESstr = str(x.AllocTRES).split(',')
        billing = [i for i in TRESstr if 'billing=' in i]
        if len(billing)==0:
            print("WARNING: Unable to find 'billing' in AllocTRES, so defaulting to NCPUS, even if this may double-count hyperthreading contributions")
            return x.NCPUS
        elif len(billing)>1:
            print("WARNING: Something weird happened, there are two 'billing' entries in AllocTRES, so defaulting to NCPUS, even if this may double-count hyperthreading contributions")
            idx = -1
        else:
            idx = billing[0].find('=')
            if idx == -1:
                print("WARNING: Unable to find 'billing' in AllocTRES, so defaulting to NCPUS, even if this may double-count hyperthreading contributions")
                return x.NCPUS
            else:
                return int(billing[0][idx+1:])

    def set_partitionType(self, x):
        assert  x in self.cluster_info['partitions'], f"\n-!- Unknown partition: {x} -!-\n"
        return self.cluster_info['partitions'][x]['type']

    def parse_timedelta(self, x):
        '''
        Parse a string representing a duration into a `datetime.timedelta` object.
        :param x: [str] Duration, as '[DD-HH:MM:]SS[.MS]'
        :return: [datetime.timedelta] Timedelta object
        '''
        # Parse number of days
        day_split = x.split('-')
        if len(day_split) == 2:
            n_days = int(day_split[0])
            HHMMSSms = day_split[1]
        else:
            n_days = 0
            HHMMSSms = x

        # Parse ms
        ms_split = HHMMSSms.split('.')
        if len(ms_split) == 2:
            n_ms = int(ms_split[1])
            HHMMSS = ms_split[0]
        else:
            n_ms = 0
            HHMMSS = HHMMSSms

        # Parse HH,MM,SS
        last_split = HHMMSS.split(':')
        if len(last_split) == 3:
            to_add = []
        elif len(last_split) == 2:
            to_add = ['00']
        elif len(last_split) == 1:
            to_add = ['00','00']
        n_h, n_m, n_s = list(map(int, to_add + last_split))

        timeD = datetime.timedelta(
            days=n_days,
            hours=n_h,
            minutes=n_m,
            seconds=n_s,
            milliseconds=n_ms
        )
        return timeD

    def calc_realMemNeeded(self, x, granularity_memory_request):
        '''
        Calculate the minimum memory needed.
        This is calculated as the smallest multiple of `granularity_memory_request` that is greater than maxRSS.
        :param x: [pd.Series] one row of sacct output.
        :param  granularity_memory_request: [float or int] level of granularity available when requesting memory on this cluster
        :return: [float] minimum memory needed, in GB.
        '''
        return min(x.ReqMemX,(int(x.UsedMem2_/granularity_memory_request)+1)*granularity_memory_request)

    def calc_CPUusage2use(self, x):
        if x.TotalCPUtime_.total_seconds() == 0: # This is when the workload manager actually didn't store real usage
            # NB: when TotalCPU=0, we assume usage factor = 100% for all CPU cores
            return x.CPUwallclocktime_
        else:
            assert x.TotalCPUtime_ <= x.CPUwallclocktime_
            return x.TotalCPUtime_

    def calc_GPUusage2use(self, x):
        if x.PartitionTypeX == 'GPU':
            if x.WallclockTimeX.total_seconds() > 0:
                assert x.NGPUS_ != 0
            return x.WallclockTimeX * x.NGPUS_ # NB assuming usage factor of 100% for GPUs
        else:
            return datetime.timedelta(0)

    def calc_coreHoursCharged(self, x,
                              # outType
                              ):

        if x.PartitionTypeX == 'CPU':
            return x.CPUwallclocktime_ / np.timedelta64(1, 'h')
        else:
            return x.WallclockTimeX * x.NGPUS_ / np.timedelta64(1, 'h')


    def clean_State(self, x, customSuccessStates_list):
        '''
        Standardise the job's state, coding with {-1,0,1}
        :param x: [str] "State" field from sacct output
        :return: [int] in [-1,0,1]
        '''
        # Codes are found here: https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES
        # self.args.customSuccessStates = 'TO,TIMEOUT'
        success_codes = ['CD','COMPLETED']
        running_codes = ['PD','PENDING','R','RUNNING','RQ','REQUEUED']
        if x in success_codes:
            codeState = 1
        elif x in customSuccessStates_list:
            # we allocate a lower value here so that when aggregating by jobID, the whole job keeps the flag
            # Otherwise a "cancelled" job could take over with StateX=0 for example
            codeState = -1
        else:
            codeState = 0

        if x in running_codes:
            # running jobs are the lowest to be removed all the time
            # (if one of the subprocess is still running, the job gets ignored regardless of --customSuccessStates
            codeState = -2

        return codeState

    def get_parent_jobID(self, x):
        '''
        Get the parent job ID in case of array jobs
        :param x: [str] JobID of the form 123456789_0 (with or without '_0')
        :return: [str] Parent ID 123456789
        '''
        foo = x.split('_')
        assert len(foo) <= 2, f"Can't parse the job ID: {x}"
        return foo[0]


class WorkloadManager(Helpers_WM):

    def __init__(self, args, cluster_info):
        '''
        Methods related to the Workload manager
        :param args: [Namespace] input from the user
        :param cluster_info: [dict] information about this specific cluster.
        '''
        self.args = args
        self.cluster_info = cluster_info
        super().__init__()

    def pull_logs(self):
        '''
        Run the command line to pull usage from the workload manager.
        More: https://slurm.schedmd.com/sacct.html
        '''
        bash_com = [
            "sacct",
            "--starttime",
            self.args.startDay,  # format YYYY-MM-DD
            "--endtime",
            self.args.endDay,  # format YYYY-MM-DD
            "--format",
            "JobID,JobName,Submit,Elapsed,Partition,NNodes,NCPUS,TotalCPU,CPUTime,ReqMem,MaxRSS,WorkDir,State,Account,AllocTres",
            "-P"
        ]

        if self.args.useCustomLogs == '':
            logs = subprocess.run(bash_com, capture_output=True)
            self.logs_raw = logs.stdout
        else:
            foo = "Overriding logs_raw with: "
            foundIt = False
            for sacctFileLocation in ['','testData','error_logs']:
                if not foundIt:
                    try:
                        with open(os.path.join(sacctFileLocation, self.args.useCustomLogs), 'rb') as f:
                            self.logs_raw = f.read()
                        foo += f"{sacctFileLocation}/{self.args.useCustomLogs}"
                        foundIt = True
                    except:
                        pass
            if not foundIt:
                raise FileNotFoundError(f"Couldn't find {self.args.useCustomLogs} \n "
                                        f"It should be either be in the testData/ or error_logs/ subdirectories, or the full path should be provided by --useCustomLogs.")
            print(foo)

    def convert2dataframe(self):
        '''
        Convert raw logs output into a pandas dataframe.
        '''
        logs_df = pd.read_csv(BytesIO(self.logs_raw), sep="|", dtype='str')
        for x in ['NNodes', 'NCPUS']:
            logs_df[x] = logs_df[x].astype('int64')

        self.logs_df = logs_df

    def clean_logs_df(self):
        '''
        Clean the different fields of the usage logs.
        NB: the name of the columns ending with X need to be conserved, as they are used by the main script.
        '''
        # self.logs_df_raw = self.logs_df.copy() # DEBUGONLY Save a copy of uncleaned raw for debugging mainly

        # Bercik Modification: Drop rows with NaN partitions since these are only 
        # administrative subjobs created by slurm and are not 'real'
        if self.cluster_info['cluster_name'] == 'Niagara':
            self.logs_df.dropna(subset=['Partition'],inplace=True)
            self.logs_df.reset_index(drop=True,inplace=True)

        # Bercik Modification: Drop rows with zero TotalCPU time and CPUTime
        if self.cluster_info['cluster_name'] == 'Niagara':
            self.logs_df.drop(self.logs_df[(self.logs_df.TotalCPU == '00:00:00') \
                                         & (self.logs_df.CPUTime == '00:00:00')].index, inplace=True)

        ### Calculate real memory usage
        self.logs_df['ReqMemX'] = self.logs_df.apply(self.calc_ReqMem, axis=1)

        ### Clean MaxRSS
        self.logs_df['UsedMem_'] = self.logs_df.apply(self.clean_RSS, axis=1)

        ### Parse wallclock time
        self.logs_df['WallclockTimeX'] = self.logs_df['Elapsed'].apply(self.parse_timedelta)

        ### Parse total CPU time
        # This is the total CPU used time, accross all cores.
        # But it is not reliably logged
        self.logs_df['TotalCPUtime_'] = self.logs_df['TotalCPU'].apply(self.parse_timedelta)

        ### Parse core-wallclock time
        # This is the maximum time cores could use, if used at 100% (Elapsed time * CPU count)
        if 'CPUTime' in self.logs_df.columns:
            self.logs_df['CPUwallclocktime_'] = self.logs_df['CPUTime'].apply(self.parse_timedelta)
        else:
            print('Using old logs, "CPUTime" information not available.') # TODO: remove this after a while
            self.logs_df['CPUwallclocktime_'] = self.logs_df.WallclockTimeX * self.logs_df.NCPUS

        ### Number of GPUs
        if 'AllocTRES' in self.logs_df.columns:
            self.logs_df['NGPUS_'] = self.logs_df.AllocTRES.str.extract(r'((?<=gres\/gpu=)\d+)', expand=False).fillna(0).astype('int64')
        else:
            print('Using old logs, "AllocTRES" information not available.')  # TODO: remove this after a while
            self.logs_df['NGPUS_'] = 0

        ### Clean partition
        # Make sure it's either a partition name, or a comma-separated list of partitions
        self.logs_df['PartitionX'] = self.logs_df.apply(self.clean_partition, axis=1)

        ### Parse submit datetime
        self.logs_df['SubmitDatetimeX'] = self.logs_df.Submit.apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S"))

        ### Number of CPUs
        # e.g. here there is no cleaning necessary, so I just standardise the column name
        if self.cluster_info['cluster_name'] == 'Niagara':
            # Bercik Modification: Need to account for hyperthreading
            self.logs_df['NCPUS_'] = self.logs_df.apply(self.clean_hyperthreading, axis=1)
        else:
            self.logs_df['NCPUS_'] = self.logs_df.NCPUS

        ### Number of nodes
        self.logs_df['NNodes_'] = self.logs_df.NNodes

        ### Job name
        self.logs_df['JobName_'] = self.logs_df.JobName

        ### Working directory
        self.logs_df['WorkingDir_'] = self.logs_df.WorkDir

        ### State
        customSuccessStates_list = self.args.customSuccessStates.split(',')
        self.logs_df['StateX'] = self.logs_df.State.apply(self.clean_State, customSuccessStates_list=customSuccessStates_list)

        ### Pull jobID
        self.logs_df['single_jobID'] = self.logs_df.JobID.apply(lambda x: x.split('.')[0])

        ### Account
        if 'Account' in self.logs_df.columns:
            self.logs_df['Account_'] = self.logs_df.Account
        else:
            print('Using old logs, "Account" information not available.') # TODO: remove this after a while
            self.logs_df['Account_'] = ''

        ### Aggregate per jobID
        self.df_agg_0 = self.logs_df.groupby('single_jobID').agg({
            'TotalCPUtime_': 'max',
            'CPUwallclocktime_': 'max',
            'WallclockTimeX': 'max',
            'ReqMemX': 'max',
            'UsedMem_': 'max',
            'NCPUS_': 'max',
            'NGPUS_': 'max',
            'NNodes_': 'max',
            'PartitionX': lambda x: ''.join(x),
            'JobName_': 'first',
            'SubmitDatetimeX': 'min',
            'WorkingDir_': 'first',
            'StateX': 'min',
            'Account_': 'first'
        })

        ### Remove jobs that are still running or currently queued
        self.df_agg = self.df_agg_0.loc[self.df_agg_0.StateX != -2]

        ### Turn StateX==-2 into 1
        self.df_agg.loc[self.df_agg.StateX == -1, 'StateX'] = 1

        ### Replace UsedMem_=-1 with memory requested (for when MaxRSS=NaN)
        self.df_agg['UsedMem2_'] = self.df_agg.apply(self.clean_UsedMem, axis=1)

        ### Label as CPU or GPU partition
        self.df_agg['PartitionTypeX'] = self.df_agg.PartitionX.apply(self.set_partitionType)

        # Just used to clean up with old logs:
        if 'AllocTRES' not in self.logs_df.columns:
            self.df_agg.loc[self.df_agg.PartitionTypeX == 'GPU','NGPUS_'] = 1 # TODO remove after a while

        # Sanity check (no GPU logged for CPU partitions and vice versa)
        assert (self.df_agg.loc[self.df_agg.PartitionTypeX == 'CPU'].NGPUS_ == 0).all()
        foo = self.df_agg.loc[(self.df_agg.PartitionTypeX == 'GPU') & (self.df_agg.NGPUS_ == 0)]
        assert (foo.WallclockTimeX.dt.total_seconds() == 0).all() # Cancelled GPU jobs won't have any GPUs allocated if they didn't start

        ### add the usage time to use for calculations
        self.df_agg['TotalCPUtime2useX'] = self.df_agg.apply(self.calc_CPUusage2use, axis=1)
        self.df_agg['TotalGPUtime2useX'] = self.df_agg.apply(self.calc_GPUusage2use, axis=1)

        ### Calculate core-hours charged
        self.df_agg['CoreHoursChargedX'] = self.df_agg.apply(self.calc_coreHoursCharged, axis=1)

        ### Calculate real memory need
        self.df_agg['NeededMemX'] = self.df_agg.apply(
            self.calc_realMemNeeded,
            granularity_memory_request=self.cluster_info['granularity_memory_request'],
            axis=1)

        ### Add memory waste information
        self.df_agg['memOverallocationFactorX'] = (self.df_agg.ReqMemX) / self.df_agg.NeededMemX

        # foo = self.df_agg[['TotalCPUtime_', 'CPUwallclocktime_', 'WallclockTimeX', 'NCPUS_', 'CoreHoursChargedCPUX',
        #                    'CoreHoursChargedGPUX', 'TotalCPUtime2useX', 'TotalGPUtime2useX']] # DEBUGONLY

        ### Filter on working directory
        if self.args.filterWD is not None:
            foo = len(self.df_agg)
            # TODO: Doesn't not work with symbolic links
            self.df_agg = self.df_agg.loc[self.df_agg.WorkingDir_ == self.args.filterWD]
            # print(f'Filtered out {foo-len(self.df_agg):,} rows (filterCWD={self.args.filterWD})') # DEBUGONLY

        ### Filter on Job ID
        self.df_agg.reset_index(inplace=True)
        self.df_agg['parentJobID'] = self.df_agg.single_jobID.apply(self.get_parent_jobID)

        if self.args.filterJobIDs != 'all':
            list_jobs2keep = self.args.filterJobIDs.split(',')
            self.df_agg = self.df_agg.loc[self.df_agg.parentJobID.isin(list_jobs2keep)]

        ### Filter on Account
        if self.args.filterAccount is not None:
            self.df_agg = self.df_agg.loc[self.df_agg.Account_ == self.args.filterAccount]
