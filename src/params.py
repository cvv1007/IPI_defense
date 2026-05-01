import argparse

def define_arguments_prompted(parser):
    parser.add_argument("--model_type", help="Type of the model to evaluate")
    parser.add_argument("--model_name", help="Name of the model to evaluate", default='', required=True)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--candidate", choices=['c0', 'c1', 'c2'], help="defense candidate", default='c0')
    parser.add_argument("--only_first_step", default=False, action='store_true', help="Only predict the first step")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Use existing dataset")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only get score without inference")
    parser.add_argument("--mini", default=False, action='store_true', help="Run a small subset")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--max_cases_dh", type=int, default=None)
    parser.add_argument("--max_cases_ds", type=int, default=None)
    parser.add_argument("--max_cases_benign", type=int, default=None)
    parser.add_argument("--shuffle", default=False, action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_benign", default=False, action='store_true')
    parser.add_argument("--max_new_tokens", type=int, default=256)

def define_arguments_finetuned(parser):
    parser.add_argument("--model_name", help="Name of the model to evaluate", required=True)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--candidate", choices=['c0', 'c1', 'c2'], help="defense candidate", default='c0')
    parser.add_argument("--only_first_step", action='store_true', help="Only predict the first step")

def parse_arguments(agent_type=None):
    parser = argparse.ArgumentParser()
    if agent_type == "prompted":
        define_arguments_prompted(parser)
    elif agent_type == "finetuned":
        define_arguments_finetuned(parser)
    args = parser.parse_args()
    return args.__dict__