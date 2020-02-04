import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--feature_extracting", type=bool, default=False, help="Training or not")
    parser.add_argument("--model_path", type=str, default="models/", help="saving model for further fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=25, help="epochs for training")
    parser.add_argument("--model_name", type=str, default='maskrcnn', help="model name to save")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for training")
    parser.add_argument("--restore", type=bool, default=False, help="Whether reload from the existing model or not")
    parser.add_argument("--dataset", type=str, default='COCO')
    args = parser.parse_args()
    return args
