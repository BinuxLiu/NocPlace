
import sys
import torch
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from datetime import datetime
import torchvision.transforms as T

import test
import utils.util as util
import utils.parser as parser
import utils.commons as commons
from losses import cosface_loss, ikt_loss
import utils.augmentations as augmentations
from nocplace_models import cosplace_network, vit_network
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset
from datasets.inherit_dataset import InheritDataset
import utils.tests as tests

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = parser.parse_arguments()
start_time = datetime.now()
args.output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
commons.make_deterministic(args.seed)
commons.setup_logging(args.output_folder, console="debug")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")

#### Model
# model = cosplace_network.GeoLocalizationNet(args.backbone, args.fc_output_dim, args.train_all_layers)
model = vit_network.GeoLocalizationNet(args.backbone)

logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

if args.resume_model is not None:
    logging.debug(f"Loading model from {args.resume_model}")
    model_state_dict = torch.load(args.resume_model)
    model.load_state_dict(model_state_dict)

model = model.to(args.device).train()

if args.use_ikt:
    train_ds = InheritDataset(args.train_set_folder, image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
    tests.inherit(args, train_ds, model)

    del train_ds

#### Optimizer
criterion = torch.nn.CrossEntropyLoss()
if args.use_ikt:
    criterion_ikt = ikt_loss.SoftTarget(args.T)
model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

#### Datasets
groups = [TrainDataset(args, args.train_set_folder + "_night", M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                       current_group=n, min_images_per_class=args.min_images_per_class) for n in range(args.groups_num)]
# Each group has its own classifier, which depends on the number of classes in the group
classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

val_ds = TestDataset(args.val_set_folder, queries_folder="queries_night",
                     positive_dist_threshold=args.positive_dist_threshold,
                     image_size=args.image_size, resize_test_imgs=args.resize_test_imgs)
logging.info(f"Validation set: {val_ds}")

#### Resume
if args.resume_train:
    model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
        util.resume_train(args, args.output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
    model = model.to(args.device)
    epoch_num = start_epoch_num - 1
    logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
else:
    best_val_recall1 = start_epoch_num = 0

if args.use_ikt:
    recalls, recalls_str = tests.test(args, val_ds, model, is_training=True)
    logging.info(f"{val_ds}: {recalls_str}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)

#### Train / evaluation loop
logging.info("Start training ...")
logging.info(f"There are {len(groups[0])} classes for the first group, " +
             f"each epoch has {args.iterations_per_epoch} iterations " +
             f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
             f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")


if args.augmentation_device == "cuda":
    gpu_augmentation = T.Compose([
            augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                    contrast=args.contrast,
                                                    saturation=args.saturation,
                                                    hue=args.hue),
            augmentations.DeviceAgnosticRandomResizedCrop([args.image_size, args.image_size],
                                                          scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

for epoch_num in range(start_epoch_num, args.epochs_num):
    
    #### Train
    epoch_start_time = datetime.now()
    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % args.groups_num
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)
    
    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)
    
    dataloader_iterator = iter(dataloader)
    model = model.train()
    
    epoch_losses = np.zeros((0, 1), dtype=np.float32)
    for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
        images, targets, _, inherited_descriptors = next(dataloader_iterator)
        images, targets, inherited_descriptors = images.to(args.device), targets.to(args.device), inherited_descriptors.to(args.device)
        
        if args.augmentation_device == "cuda":
            images = gpu_augmentation(images)
        
        model_optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()
        
        if not args.use_amp16:
            descriptors = model(images)
            output = classifiers[current_group_num](descriptors, targets)
            loss = criterion(output, targets)
            if args.use_ikt:
                inherited_output = classifiers[current_group_num](inherited_descriptors, targets)
                loss_ikt = criterion_ikt(output, inherited_output) * args.lambda_ikt
                loss = loss + loss_ikt
            loss.backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images, inherited_descriptors
            model_optimizer.step()
            classifiers_optimizers[current_group_num].step()
        else:  # Use AMP 16
            with torch.cuda.amp.autocast():
                descriptors = model(images)
                output = classifiers[current_group_num](descriptors, targets)
                loss = criterion(output, targets)
            scaler.scale(loss).backward()
            epoch_losses = np.append(epoch_losses, loss.item())
            del loss, output, images
            scaler.step(model_optimizer)
            scaler.step(classifiers_optimizers[current_group_num])
            scaler.update()
    
    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
    
    logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {epoch_losses.mean():.4f}")
    
    #### Evaluation
    recalls, recalls_str = tests.test(args, val_ds, model, is_training=True)
    logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
    is_best = recalls[0] > best_val_recall1
    best_val_recall1 = max(recalls[0], best_val_recall1)
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint({
        "epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model_optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "best_val_recall1": best_val_recall1
    }, is_best, args.output_folder)


logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

logging.info("Experiment finished (without any errors)")