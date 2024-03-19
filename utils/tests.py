import os
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset

import utils.visualizations as visualizations


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


def test_efficient_ram_usage(args: Namespace, eval_ds: Dataset, model: torch.nn.Module,
                             num_preds_to_save: int = 0) -> Tuple[np.ndarray, str]:
    
    model = model.eval()
    distances = np.empty([eval_ds.queries_num, eval_ds.database_num], dtype=np.float32)
    
    with torch.no_grad():
        queries_features = np.ones((eval_ds.queries_num, args.fc_output_dim), dtype="float32")
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            features = model(images.to(args.device))
            queries_features[indices.numpy()-eval_ds.database_num, :] = features.cpu().numpy()

        queries_features = torch.tensor(queries_features).type(torch.float32).cuda()

        logging.debug("Extracting database features for evaluation/testing")

        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            inputs = inputs.to(args.device)
            features = model(inputs)
            for pn, (index, pred_feature) in enumerate(zip(indices, features)):
                distances[:, index] = ((queries_features-pred_feature)**2).sum(1).cpu().numpy()
        del features, queries_features, pred_feature

    predictions = distances.argsort(axis=1)[:, :max(RECALL_VALUES)]
    del distances

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    
    # Save visualizations of predictions
    if num_preds_to_save != 0:
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, args.output_folder, args.save_only_wrong_preds)

    return recalls, recalls_str


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module,
         num_preds_to_save: int = 0) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""
    
    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model)

    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")

        database_descriptors_dir = os.path.join(eval_ds.dataset_folder, f"database_{args.backbone}_{args.fc_output_dim}.npy")
        if os.path.isfile(database_descriptors_dir) == 1:
            database_descriptors = np.load(database_descriptors_dir)
        else: 
            for images, indices in tqdm(database_dataloader, ncols=100):
                descriptors = model(images.to(args.device))
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors

            database_descriptors = all_descriptors[:eval_ds.database_num]
            np.save(database_descriptors_dir, database_descriptors)
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]
    
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    
    # Save visualizations of predictions
    if num_preds_to_save != 0:
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, args.output_folder, args.save_only_wrong_preds)
    
    return recalls, recalls_str


def test_tokyo(args: Namespace, eval_ds: Dataset, model: torch.nn.Module,
               num_preds_to_save: int = 0) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")

        database_descriptors_dir = os.path.join(eval_ds.dataset_folder, f"database_{args.backbone}_{args.fc_output_dim}.npy")
        if os.path.isfile(database_descriptors_dir) == 1:
            database_descriptors = np.load(database_descriptors_dir)
        else: 
            for images, indices in tqdm(database_dataloader, ncols=100):
                descriptors = model(images.to(args.device))
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors

            database_descriptors = all_descriptors[:eval_ds.database_num]
            np.save(database_descriptors_dir, database_descriptors)
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]
    
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    recalls_day = np.zeros(len(RECALL_VALUES))
    recalls_sunset = np.zeros(len(RECALL_VALUES))
    recalls_night = np.zeros(len(RECALL_VALUES))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                if query_index%3 == 0:
                    recalls_day[i:] += 1
                elif query_index%3 == 1:
                    recalls_sunset[i:] += 1
                elif query_index%3 == 2:
                    recalls_night[i:] += 1
                recalls[i:] += 1
                break
    
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_day = recalls_day / eval_ds.queries_num * 100 * 3
    recalls_sunset = recalls_sunset / eval_ds.queries_num * 100 * 3
    recalls_night = recalls_night / eval_ds.queries_num * 100 * 3
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    recalls_str_day = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls_day)])
    recalls_str_sunset = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls_sunset)])
    recalls_str_night = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls_night)])
    
    # Save visualizations of predictions
    if num_preds_to_save != 0:
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :num_preds_to_save], eval_ds, args.output_folder, args.save_only_wrong_preds)
    
    return recalls, recalls_str,  recalls_str_day, recalls_str_sunset, recalls_str_night


def inherit(args: Namespace, eval_ds: Dataset, model: torch.nn.Module):
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for IKT")
        dataloader = DataLoader(dataset=eval_ds, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, image_paths, indices in tqdm(dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            for idx, image_path in enumerate(image_paths):
                np.save(image_path.replace("train", "train_feat").replace(".jpg", ".npy"), descriptors[idx])