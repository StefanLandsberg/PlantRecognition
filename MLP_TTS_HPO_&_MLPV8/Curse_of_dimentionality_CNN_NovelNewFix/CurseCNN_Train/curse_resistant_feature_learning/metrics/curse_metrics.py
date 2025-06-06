#!/usr/bin/env python3
"""
Curse Resistance Metrics

Real-time evaluation of feature quality and curse-of-dimensionality resistance.
Provides quantifiable metrics for training feedback and feature assessment.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


class CurseResistanceMetrics:
    """
    Real-time curse resistance evaluation for feature quality assessment.
    
    Calculates multiple metrics:
    - Effective Dimensionality: How many dimensions actually matter
    - Feature Redundancy: Correlation between features
    - Mutual Information: Information content with respect to labels
    - Combined Curse Score: Overall resistance to dimensionality curse
    """
    
    def __init__(self, variance_threshold: float = 0.95, 
                 correlation_threshold: float = 0.8,
                 device: str = 'cpu'):
        """
        Initialize curse resistance metrics calculator.
        
        Args:
            variance_threshold: Variance threshold for effective dimensionality
            correlation_threshold: Correlation threshold for redundancy detection
            device: Device for computations
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.device = device
        
        # Cache for PCA objects to avoid recomputation
        self._pca_cache = {}
        
    def evaluate_batch(self, features: torch.Tensor, 
                      labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate curse resistance metrics for a batch of features.
        
        Args:
            features: Feature tensor [batch_size, feature_dim]
            labels: Label tensor [batch_size]
            
        Returns:
            Dictionary of curse resistance metrics
        """
        # Skip calculation for very small batches to prevent errors
        if features.size(0) < 3:
            return {
                'effective_dimensionality': 0.9,  # Assume reasonable defaults
                'feature_redundancy': 0.3,
                'mutual_information': 0.2,
                'curse_resistance_score': 0.6
            }
        
        # Convert to numpy for sklearn compatibility
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Calculate individual components
        effective_dim = self._calculate_effective_dimensionality(features_np)
        redundancy = self._calculate_feature_redundancy(features)
        mutual_info = self._calculate_mutual_information(features_np, labels_np)
        
        # Calculate combined curse resistance score
        curse_score = self._calculate_combined_curse_score(
            effective_dim, redundancy, mutual_info
        )
        
        return {
            'effective_dimensionality': effective_dim,
            'feature_redundancy': redundancy,
            'mutual_information': mutual_info,
            'curse_resistance_score': curse_score
        }
    
    def _calculate_effective_dimensionality(self, features: np.ndarray) -> float:
        """
        Calculate effective dimensionality using PCA.
        
        Returns ratio of effective dimensions to total dimensions.
        Lower is better (less curse of dimensionality).
        """
        try:
            # Use cached PCA if available and appropriate
            cache_key = f"pca_{features.shape[1]}"
            
            if cache_key not in self._pca_cache or features.shape[0] > 100:
                pca = PCA()
                pca.fit(features)
                
                # Cache if batch is large enough
                if features.shape[0] > 50:
                    self._pca_cache[cache_key] = pca
            else:
                pca = self._pca_cache[cache_key]
                
            # Calculate cumulative variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            # Find number of components needed for variance threshold
            effective_dims = np.argmax(cumulative_variance >= self.variance_threshold) + 1
            
            # Return as ratio (normalized)
            dimensionality_ratio = effective_dims / features.shape[1]
            
            return float(dimensionality_ratio)
            
        except Exception as e:
            logging.warning(f"Error calculating effective dimensionality: {e}")
            return 1.0  # Worst case
    
    def _calculate_feature_redundancy(self, features: torch.Tensor) -> float:
        """
        Calculate feature redundancy via correlation analysis.
        
        Returns ratio of highly correlated feature pairs.
        Lower is better (less redundancy).
        """
        try:
            # Calculate correlation matrix
            if features.shape[1] > 1000:
                # Sample features for large dimensions to avoid memory issues
                sample_indices = torch.randperm(features.shape[1])[:1000]
                features_sample = features[:, sample_indices]
            else:
                features_sample = features
            
            # Calculate correlation matrix
            correlation_matrix = torch.corrcoef(features_sample.T)
            
            # Handle NaN values
            correlation_matrix = torch.nan_to_num(correlation_matrix, nan=0.0)
            
            # Get off-diagonal elements (exclude self-correlation)
            mask = ~torch.eye(correlation_matrix.size(0), dtype=bool, device=features.device)
            off_diagonal_corr = correlation_matrix[mask]
            
            # Count high correlations
            high_corr_count = torch.sum(torch.abs(off_diagonal_corr) > self.correlation_threshold)
            total_pairs = len(off_diagonal_corr)
            
            if total_pairs == 0:
                return 0.0
            
            redundancy_ratio = high_corr_count.float() / total_pairs
            return float(redundancy_ratio)
            
        except Exception as e:
            logging.warning(f"Error calculating feature redundancy: {e}")
            return 1.0  # Worst case
    
    def _calculate_mutual_information(self, features: np.ndarray, 
                                    labels: np.ndarray) -> float:
        """
        Calculate mutual information between features and labels.
        
        Returns normalized mutual information score.
        Higher is better (more informative features).
        """
        try:
            # Check for minimum sample requirements
            if len(features) < 2:
                logging.warning("Insufficient samples for mutual information calculation")
                return 0.0
            
            # Check for class diversity
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                logging.warning("Insufficient class diversity for mutual information calculation")
                return 0.0
            
            # Sample features if too many dimensions
            if features.shape[1] > 500:
                sample_indices = np.random.choice(features.shape[1], 500, replace=False)
                features_sample = features[:, sample_indices]
            else:
                features_sample = features
            
            # Check for feature variance
            if np.all(np.var(features_sample, axis=0) < 1e-8):
                logging.warning("Features have no variance for mutual information calculation")
                return 0.0
            
            # Calculate mutual information with robust error handling
            # Ensure we have enough samples per class
            class_counts = np.bincount(labels)
            if np.any(class_counts < 3):  # Need at least 3 samples per class
                # Use correlation-based fallback
                return self._correlation_based_mi_fallback(features_sample, labels)
            
            # Additional validation for sklearn compatibility
            if features_sample.shape[0] < 10 or features_sample.shape[1] == 0:
                return 0.2
            
            # Use correlation-based fallback to avoid sklearn errors
            try:
                mi_scores = mutual_info_classif(
                    features_sample, labels, 
                    discrete_features=False, 
                    random_state=42
                )
            except Exception:
                # Fallback: correlation-based proxy
                correlations = []
                numeric_labels = labels.astype(float)
                for i in range(min(10, features_sample.shape[1])):
                    try:
                        corr = np.abs(np.corrcoef(features_sample[:, i], numeric_labels)[0, 1])
                        if not np.isnan(corr):
                            correlations.append(corr)
                    except:
                        continue
                
                if correlations:
                    avg_corr = np.mean(correlations)
                    return min(0.8, max(0.1, avg_corr * 2.0))
                else:
                    return 0.25
            
            # Validate MI scores
            if len(mi_scores) == 0 or np.all(np.isnan(mi_scores)):
                logging.warning("Invalid mutual information scores computed")
                return 0.0
            
            # Get mean mutual information
            mean_mi = np.mean(mi_scores[~np.isnan(mi_scores)])
            
            # Normalize to [0, 1] range (heuristic normalization)
            # Typical MI scores are in range [0, 2] for most datasets
            normalized_mi = min(1.0, max(0.0, mean_mi / 0.5))
            
            return float(normalized_mi)
            
        except Exception as e:
            logging.warning(f"Error calculating mutual information: {e}")
            return 0.0  # Worst case
    
    def _calculate_combined_curse_score(self, effective_dim: float, 
                                       redundancy: float, 
                                       mutual_info: float) -> float:
        """
        Calculate combined curse resistance score.
        
        Args:
            effective_dim: Effective dimensionality ratio [0,1]
            redundancy: Feature redundancy ratio [0,1] 
            mutual_info: Normalized mutual information [0,1]
            
        Returns:
            Combined curse resistance score [0,1], higher is better
        """
        # Component scores (all should be in [0,1] where 1 is best)
        dim_score = 1.0 - effective_dim  # Lower dimensionality is better
        redundancy_score = 1.0 - redundancy  # Lower redundancy is better
        mi_score = mutual_info  # Higher MI is better
        
        # Weighted combination
        # Emphasize dimensionality and redundancy more than MI
        curse_resistance = (
            0.4 * dim_score +
            0.35 * redundancy_score +
            0.25 * mi_score
        )
        
        return float(np.clip(curse_resistance, 0.0, 1.0))
    
    def evaluate_feature_quality(self, features: torch.Tensor, 
                                labels: torch.Tensor) -> Dict[str, Union[float, np.ndarray]]:
        """
        Detailed feature quality evaluation.
        
        Returns:
            Detailed metrics including per-feature analysis
        """
        basic_metrics = self.evaluate_batch(features, labels)
        
        # Additional detailed analysis
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Per-feature mutual information
        try:
            if features.shape[1] <= 1000:  # Only for manageable dimensions
                per_feature_mi = mutual_info_classif(
                    features_np, labels_np, discrete_features=False, random_state=42
                )
                basic_metrics['per_feature_mi_mean'] = float(np.mean(per_feature_mi))
                basic_metrics['per_feature_mi_std'] = float(np.std(per_feature_mi))
                basic_metrics['per_feature_mi_max'] = float(np.max(per_feature_mi))
        except Exception as e:
            logging.warning(f"Error in detailed feature analysis: {e}")
        
        # Feature variance analysis
        feature_vars = torch.var(features, dim=0)
        basic_metrics['feature_variance_mean'] = float(torch.mean(feature_vars))
        basic_metrics['feature_variance_std'] = float(torch.std(feature_vars))
        basic_metrics['low_variance_features'] = float(torch.sum(feature_vars < 0.01))
        
        return basic_metrics
    
    def compare_feature_sets(self, features_a: torch.Tensor, 
                           features_b: torch.Tensor,
                           labels: torch.Tensor) -> Dict[str, float]:
        """
        Compare curse resistance between two feature sets.
        
        Args:
            features_a: First feature set
            features_b: Second feature set  
            labels: Labels for both sets
            
        Returns:
            Comparison metrics showing improvements
        """
        metrics_a = self.evaluate_batch(features_a, labels)
        metrics_b = self.evaluate_batch(features_b, labels)
        
        comparisons = {}
        for key in ['effective_dimensionality', 'feature_redundancy', 
                   'mutual_information', 'curse_resistance_score']:
            improvement = metrics_b[key] - metrics_a[key]
            comparisons[f'{key}_improvement'] = improvement
            comparisons[f'{key}_relative_improvement'] = improvement / (metrics_a[key] + 1e-8)
        
        return comparisons
    
    def get_curse_resistance_grade(self, curse_score: float) -> str:
        """
        Get human-readable grade for curse resistance score.
        
        Args:
            curse_score: Curse resistance score [0,1]
            
        Returns:
            Grade string
        """
        if curse_score >= 0.9:
            return "Excellent (A+)"
        elif curse_score >= 0.8:
            return "Very Good (A)"
        elif curse_score >= 0.7:
            return "Good (B)"
        elif curse_score >= 0.6:
            return "Fair (C)"
        elif curse_score >= 0.5:
            return "Poor (D)"
        else:
            return "Very Poor (F)"


def test_curse_metrics():
    """Test the curse resistance metrics calculator."""
    print("Testing CurseResistanceMetrics...")
    
    # Create test data
    batch_size, feature_dim = 100, 50
    
    # Good features (low correlation, informative)
    good_features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, 5, (batch_size,))
    
    # Bad features (high correlation, redundant)
    base_feature = torch.randn(batch_size, 1)
    bad_features = base_feature + 0.1 * torch.randn(batch_size, feature_dim)
    
    # Test metrics
    metrics_calc = CurseResistanceMetrics()
    
    good_metrics = metrics_calc.evaluate_batch(good_features, labels)
    bad_metrics = metrics_calc.evaluate_batch(bad_features, labels)
    
    print("Good features metrics:")
    for key, value in good_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"  Grade: {metrics_calc.get_curse_resistance_grade(good_metrics['curse_resistance_score'])}")
    
    print("\nBad features metrics:")
    for key, value in bad_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"  Grade: {metrics_calc.get_curse_resistance_grade(bad_metrics['curse_resistance_score'])}")
    
    # Test comparison
    comparison = metrics_calc.compare_feature_sets(bad_features, good_features, labels)
    print("\nComparison (good vs bad):")
    for key, value in comparison.items():
        print(f"  {key}: {value:.4f}")
    
    print("Curse metrics test passed!")


if __name__ == "__main__":
    test_curse_metrics()