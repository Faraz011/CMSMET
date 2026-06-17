"""
Evaluation utilities for CMSMET
Cross-modal transfer evaluation and metrics
"""
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import euclidean, cosine
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

from cmsmet.models.encoders import SharedEmotionSpace


class CrossModalEvaluator:
    """Evaluate cross-modal emotion transfer"""
    
    def __init__(
        self,
        model: SharedEmotionSpace,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            model: Trained SharedEmotionSpace model
            device: Device to use
            logger: Logger instance
        """
        self.model = model
        self.device = torch.device(device)
        self.logger = logger or logging.getLogger("evaluator")
        
        self.model.eval()
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader,
        modality: str = "speech",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from data
        
        Args:
            dataloader: Data loader
            modality: 'speech' or 'music'
        
        Returns:
            embeddings: (n_samples, embedding_dim)
            labels: (n_samples,)
        """
        embeddings_list = []
        labels_list = []
        
        for batch in dataloader:
            waveform = batch['waveform'].to(self.device)
            emotion = batch['emotion'].numpy() if isinstance(batch['emotion'], torch.Tensor) else batch['emotion']
            
            if modality == "speech":
                emb, _ = self.model.forward_speech(waveform)
            else:
                emb, _ = self.model.forward_music(waveform)
            
            embeddings_list.append(emb.cpu().numpy())
            if emotion.ndim == 1:
                labels_list.append(emotion)
            else:
                labels_list.append(emotion.argmax(axis=1))
        
        embeddings = np.concatenate(embeddings_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return embeddings, labels
    
    def compute_embedding_similarity(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        labels_a: np.ndarray,
        labels_b: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute similarity between embeddings from different modalities
        
        Args:
            embeddings_a: Embeddings from modality A (n_samples, embedding_dim)
            embeddings_b: Embeddings from modality B (n_samples, embedding_dim)
            labels_a: Emotion labels for A
            labels_b: Emotion labels for B
        
        Returns:
            Dict with similarity metrics
        """
        metrics = {}
        
        # Cosine similarity between corresponding samples
        cosine_sims = []
        for i in range(len(embeddings_a)):
            sim = 1 - cosine(embeddings_a[i], embeddings_b[i])
            cosine_sims.append(sim)
        
        metrics['mean_cosine_similarity'] = np.mean(cosine_sims)
        metrics['std_cosine_similarity'] = np.std(cosine_sims)
        metrics['min_cosine_similarity'] = np.min(cosine_sims)
        metrics['max_cosine_similarity'] = np.max(cosine_sims)
        
        # Correlation between similarity and label match
        label_match = (labels_a == labels_b).astype(float)
        corr, pval = pearsonr(cosine_sims, label_match)
        metrics['similarity_label_corr'] = corr
        metrics['similarity_label_pval'] = pval
        
        return metrics
    
    def probe_classifier(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
        train_source: str = "speech",
        test_source: str = "music",
    ) -> Dict[str, float]:
        """
        Probe linear classifier for cross-modal transfer
        
        Args:
            train_embeddings: Training embeddings (n_train, embedding_dim)
            train_labels: Training labels (n_train,)
            test_embeddings: Test embeddings (n_test, embedding_dim)
            test_labels: Test labels (n_test,)
            train_source: Source modality for training ('speech' or 'music')
            test_source: Source modality for testing
        
        Returns:
            Dict with probe classifier metrics
        """
        # Handle multi-dimensional emotions (reduce to single class)
        if train_labels.ndim > 1:
            train_labels = train_labels.argmax(axis=1)
        if test_labels.ndim > 1:
            test_labels = test_labels.argmax(axis=1)
        
        # Standardize
        scaler = StandardScaler()
        train_emb_scaled = scaler.fit_transform(train_embeddings)
        test_emb_scaled = scaler.transform(test_embeddings)
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_emb_scaled, train_labels)
        
        # Evaluate
        train_pred = clf.predict(train_emb_scaled)
        test_pred = clf.predict(test_emb_scaled)
        
        train_acc = accuracy_score(train_labels, train_pred)
        test_acc = accuracy_score(test_labels, test_pred)
        
        train_f1 = f1_score(train_labels, train_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(test_labels, test_pred, average='weighted', zero_division=0)
        
        return {
            f'{train_source}_to_{test_source}_train_acc': train_acc,
            f'{train_source}_to_{test_source}_test_acc': test_acc,
            f'{train_source}_to_{test_source}_train_f1': train_f1,
            f'{train_source}_to_{test_source}_test_f1': test_f1,
        }
    
    def evaluate_single_modal_baseline(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        modality: str = "speech",
        split_ratio: float = 0.8,
    ) -> Dict[str, float]:
        """
        Evaluate single-modal baseline (no cross-modal transfer)
        
        Args:
            embeddings: Embeddings (n_samples, embedding_dim)
            labels: Labels (n_samples,)
            modality: Modality name
            split_ratio: Train/test split ratio
        
        Returns:
            Dict with metrics
        """
        # Split data
        n_train = int(len(embeddings) * split_ratio)
        indices = np.random.permutation(len(embeddings))
        
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_emb = embeddings[train_idx]
        train_labels = labels[train_idx]
        test_emb = embeddings[test_idx]
        test_labels = labels[test_idx]
        
        if train_labels.ndim > 1:
            train_labels = train_labels.argmax(axis=1)
        if test_labels.ndim > 1:
            test_labels = test_labels.argmax(axis=1)
        
        # Standardize
        scaler = StandardScaler()
        train_emb_scaled = scaler.fit_transform(train_emb)
        test_emb_scaled = scaler.transform(test_emb)
        
        # Classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_emb_scaled, train_labels)
        
        # Metrics
        train_pred = clf.predict(train_emb_scaled)
        test_pred = clf.predict(test_emb_scaled)
        
        return {
            f'{modality}_baseline_train_acc': accuracy_score(train_labels, train_pred),
            f'{modality}_baseline_test_acc': accuracy_score(test_labels, test_pred),
            f'{modality}_baseline_train_f1': f1_score(train_labels, train_pred, average='weighted', zero_division=0),
            f'{modality}_baseline_test_f1': f1_score(test_labels, test_pred, average='weighted', zero_division=0),
        }
    
    def compute_emotion_correlation(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute correlation between modalities for emotion recognition
        
        Args:
            embeddings_a: First modality embeddings
            embeddings_b: Second modality embeddings
            labels: Emotion labels
        
        Returns:
            Dict with correlation metrics
        """
        metrics = {}
        
        # Group by emotion
        unique_labels = np.unique(labels)
        
        for emotion_id in unique_labels:
            mask = labels == emotion_id
            if mask.sum() < 2:
                continue
            
            emb_a = embeddings_a[mask]
            emb_b = embeddings_b[mask]
            
            # Compute mean embedding for each emotion
            mean_a = emb_a.mean(axis=0)
            mean_b = emb_b.mean(axis=0)
            
            # Cosine similarity
            sim = 1 - cosine(mean_a, mean_b)
            metrics[f'emotion_{emotion_id}_similarity'] = sim
        
        if metrics:
            metrics['mean_emotion_similarity'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def evaluate_all(
        self,
        speech_train_loader,
        music_train_loader,
        speech_test_loader,
        music_test_loader,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run comprehensive evaluation
        
        Args:
            speech_train_loader: Speech training data
            music_train_loader: Music training data
            speech_test_loader: Speech test data
            music_test_loader: Music test data
            output_dir: Directory to save results
        
        Returns:
            Dict with all evaluation metrics
        """
        self.logger.info("Extracting embeddings...")
        
        speech_train_emb, speech_train_labels = self.extract_embeddings(
            speech_train_loader, "speech"
        )
        speech_test_emb, speech_test_labels = self.extract_embeddings(
            speech_test_loader, "speech"
        )
        music_train_emb, music_train_labels = self.extract_embeddings(
            music_train_loader, "music"
        )
        music_test_emb, music_test_labels = self.extract_embeddings(
            music_test_loader, "music"
        )
        
        all_metrics = {}
        
        # Baseline evaluations
        self.logger.info("Evaluating single-modal baselines...")
        baseline_speech = self.evaluate_single_modal_baseline(
            speech_train_emb, speech_train_labels, "speech"
        )
        baseline_music = self.evaluate_single_modal_baseline(
            music_train_emb, music_train_labels, "music"
        )
        all_metrics.update(baseline_speech)
        all_metrics.update(baseline_music)
        
        # Cross-modal transfer
        self.logger.info("Evaluating cross-modal transfer...")
        transfer_s2m = self.probe_classifier(
            speech_train_emb, speech_train_labels,
            music_test_emb, music_test_labels,
            "speech", "music"
        )
        transfer_m2s = self.probe_classifier(
            music_train_emb, music_train_labels,
            speech_test_emb, speech_test_labels,
            "music", "speech"
        )
        all_metrics.update(transfer_s2m)
        all_metrics.update(transfer_m2s)
        
        # Embedding similarity
        self.logger.info("Computing embedding similarities...")
        similarity = self.compute_embedding_similarity(
            speech_test_emb, music_test_emb,
            speech_test_labels, music_test_labels
        )
        all_metrics.update(similarity)
        
        # Emotion correlations
        emotion_corr = self.compute_emotion_correlation(
            speech_test_emb, music_test_emb,
            speech_test_labels
        )
        all_metrics.update(emotion_corr)
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            import json
            with open(os.path.join(output_dir, "evaluation_metrics.json"), 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_metrics = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in all_metrics.items()
                }
                json.dump(json_metrics, f, indent=2)
        
        return all_metrics
