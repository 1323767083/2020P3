import sys

class main():
    def __init__(self, argv):
        pass

    def run(self):
        pass


if __name__ == '__main__':
    main(sys.argv[1:]).run()

'''
#To introduce a new data source.

step 1: extract data
    File:   data_prepare_qz.py
    sample: python data_prepare_qz.py extract_20180101_qz_data

step 2: intergrate in the system
    1.update API_qz_from_file in data_common.py
    sample   check _V1_ and _V2_ tag
    2.update API_qz_data_source_related in data_common.py three methods
    
    
    3.update global variable hfq_src_base_dir in data_common.py 
    4.update global variable index_src_base_dir in data_common.py


step 3: prepare intermidate data
    intermediate data inlude: summary data, addon data and index data
    in the current setup T5, only summary data and add on data is needed
    
    intermediate data sample : 
    python data_intermediate_result.py prepare_intermediate_summary_data
    python data_intermediate_result.py prepare_intermediate_addon_data 


step 4: prepare data
    python data_T5.py
'''