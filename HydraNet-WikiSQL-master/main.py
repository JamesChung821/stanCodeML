import argparse
import os
import sys
import shutil
import datetime

import matplotlib.pyplot as plt
import numpy as np

import utils
from modeling.model_factory import create_model
from featurizer import HydraFeaturizer, SQLDataset
from evaluator import HydraEvaluator
import torch.utils.data as torch_data

parser = argparse.ArgumentParser(description='HydraNet training script')
parser.add_argument("job", type=str, choices=["train"],
                    help="job can be train")
parser.add_argument("--conf", help="conf file path")
parser.add_argument("--output_path", type=str, default="output", help="folder path for all outputs")
parser.add_argument("--model_path", help="trained model folder path (used in eval, predict and export mode)")
parser.add_argument("--epoch", help="epochs to restore (used in eval, predict and export mode)")
parser.add_argument("--gpu", type=str, default=None, help="gpu id")
parser.add_argument("--note", type=str)

args = parser.parse_args()




if args.job == "train":
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    conf_path = os.path.abspath(args.conf)
    config = utils.read_conf(conf_path)

    note = args.note if args.note else ""

    script_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    output_path = args.output_path
    model_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_path, model_name)

    # Load the general checkpoint
    checkpoint_model_path = "best_model"
    checkpoint_epoch = int(config["checkpoint_epoch"])   # 4

    if "DEBUG" not in config:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        shutil.copyfile(conf_path, os.path.join(model_path, "model.conf"))
        shutil.copytree("modeling", os.path.join(model_path, "modeling"))   # <--------------- newly added
        for pyfile in ["featurizer.py", "utils.py"]:    # add utils.py
            shutil.copyfile(pyfile, os.path.join(model_path, pyfile))
        if config["model_type"] == "pytorch":
            shutil.copyfile("modeling/torch_model.py", os.path.join(model_path, "torch_model.py"))
        elif config["model_type"] == "tf":
            shutil.copyfile("modeling/tf_model.py", os.path.join(model_path, "tf_model.py"))
        else:
            raise Exception("model_type is not supported")

    featurizer = HydraFeaturizer(config)
    train_data = SQLDataset(config["train_data_path"], config, featurizer, True)
    train_data_loader = torch_data.DataLoader(train_data, batch_size=int(config["batch_size"]), shuffle=True, pin_memory=True)

    num_samples = len(train_data)
    config["num_train_steps"] = int(num_samples * int(config["epochs"]) / int(config["batch_size"]))
    step_per_epoch = num_samples / int(config["batch_size"])
    print("total_steps: {0}, warm_up_steps: {1}".format(config["num_train_steps"], config["num_warmup_steps"]))

    model = create_model(config, is_train=True)
    if config["load_checkpoint"] == 'True':
        model.load(checkpoint_model_path, checkpoint_epoch)     # <--------------- added to load checkpoint best model
    evaluator = HydraEvaluator(model_path, config, featurizer, model, note)
    print("start training")
    loss_avg, step, epoch = 0.0, 0, 0
    print('This is train_data_loader:', len(train_data_loader))
    while True:
        history = {'cur_loss': np.array([]), 'loss_avg': np.array([]), 'batch_id': np.array([])}    # <--- added to plot
        for batch_id, batch in enumerate(train_data_loader):
            # print(batch_id)
            cur_loss = model.train_on_batch(batch)
            loss_avg = (loss_avg * step + cur_loss) / (step + 1)
            history['cur_loss'] = np.append(history['cur_loss'], cur_loss)  # <--- added to plot
            history['loss_avg'] = np.append(history['loss_avg'], loss_avg)  # <--- added to plot
            history['batch_id'] = np.append(history['batch_id'], batch_id)  # <--- added to plot
            step += 1
            if batch_id % 50 == 0:
                model.save_iteration(model_path, epoch, batch_id, cur_loss)   # <--------------------------- added
                currentDT = datetime.datetime.now()
                print("[{3}] epoch {0}, batch {1}, batch_loss={2:.4f}".format(epoch, batch_id, cur_loss,
                                                                                currentDT.strftime("%m-%d %H:%M:%S")))
        if args.note:
            print(args.note)
        model.save(model_path, epoch)
        # evaluator.eval(epoch)

        fig = plt.figure()
        plt.plot(history['batch_id'], history['cur_loss'], 'b-', label='Current loss')  # <--- added to plot
        plt.plot(history['batch_id'], history['loss_avg'], 'k--', label='Average loss')  # <--- added to plot
        plt.legend()  # <--- added to plot
        plt.xlabel('Batch ID')  # <--- added to plot
        plt.ylabel('Loss')  # <--- added to plot
        plt.title(f'Model_{epoch} Loss history')  # <--- added to plot
        fig.savefig("{1}/model_{0}_loss.png".format(epoch, model_path))  # <--- added to save figure
        # plt.show()  # <--- added to plot

        epoch += 1
        if epoch >= int(config["epochs"]):
            break
else:
    raise Exception("Job type {0} is not supported for now".format(args.job))