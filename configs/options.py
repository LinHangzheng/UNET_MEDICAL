import argparse
### Default CLI options
# Apps should use these CLI options, and then 
# extend using parser.add_argument_group('app')

def parse_options():
    parser = argparse.ArgumentParser(description='Deep learning model for cell image segmentation.')
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--configs',type=str, default='configs/params.yaml',
                             help='path to the parameters.')
    args = parser.parse_args()
    return args