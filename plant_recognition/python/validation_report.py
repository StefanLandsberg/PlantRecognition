import json
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class MLPipelineValidator:
    """Validation system for the two-stage ML pipeline."""
    
    def __init__(self, config_path="../models/ml_config.json"):
        self.config = self.load_config(config_path)
        self.results = {}
        
    def load_config(self, config_path):
        """Load validation configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            return {
                "validation_data_path": "../data/validation/",
                "test_data_path": "../data/test/",
                "output_path": "../reports/",
                "generate_plots": True
            }
    
    def validate_stage_a(self, model, test_data):
        """Validate Stage A: Binary detection."""
        print("Validating Stage A: Binary Detection...")
        
        correct = 0
        total = 0
        predictions = []
        ground_truth = []
        
        for image_path, label in test_data:
            try:
                # Preprocess image
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted = probabilities[0, 1].item() > self.config.get('binary_threshold', 0.5)
                
                predictions.append(predicted)
                ground_truth.append(label == 'invasive')
                
                if predicted == (label == 'invasive'):
                    correct += 1
                total += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        
        self.results['stage_a'] = {
            'accuracy': accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'predictions': predictions,
            'ground_truth': ground_truth
        }
        
        print(f"Stage A Accuracy: {accuracy:.4f}")
        return accuracy
    
    def validate_stage_b(self, pipeline, test_data):
        """Validate Stage B: Fine-grained classification."""
        print("Validating Stage B: Fine-grained Classification...")
        
        correct = 0
        total = 0
        unknown_detected = 0
        predictions = []
        ground_truth = []
        confidence_scores = []
        
        for image_path, label in test_data:
            try:
                # Preprocess image
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                image = Image.open(image_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                
                # Predict using pipeline
                result = pipeline.predict(image_tensor)
                
                predicted_species = result['predicted_species']
                confidence = result['confidence']
                is_unknown = result.get('is_unknown', False)
                
                predictions.append(predicted_species)
                ground_truth.append(label)
                confidence_scores.append(confidence)
                
                if is_unknown:
                    unknown_detected += 1
                elif predicted_species == label:
                    correct += 1
                
                total += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        unknown_rate = unknown_detected / total if total > 0 else 0
        
        self.results['stage_b'] = {
            'accuracy': accuracy,
            'unknown_detection_rate': unknown_rate,
            'total_samples': total,
            'correct_predictions': correct,
            'unknown_detected': unknown_detected,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'confidence_scores': confidence_scores
        }
        
        print(f"Stage B Accuracy: {accuracy:.4f}")
        print(f"Unknown Detection Rate: {unknown_rate:.4f}")
        return accuracy, unknown_rate
    
    def generate_confusion_matrix(self, predictions, ground_truth, stage_name):
        """Generate confusion matrix."""
        try:
            cm = confusion_matrix(ground_truth, predictions)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {stage_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            output_path = f"{self.config.get('output_path', '../reports/')}confusion_matrix_{stage_name.lower()}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            return None
    
    def generate_confidence_distribution(self, confidence_scores, stage_name):
        """Generate confidence score distribution plot."""
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(confidence_scores, bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'Confidence Score Distribution - {stage_name}')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.axvline(x=self.config.get('min_confidence', 0.6), color='red', linestyle='--', label='Threshold')
            plt.legend()
            
            output_path = f"{self.config.get('output_path', '../reports/')}confidence_distribution_{stage_name.lower()}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Error generating confidence distribution: {e}")
            return None
    
    def generate_report(self):
        """Generate comprehensive validation report."""
        report = {
            "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_version": "2.0",
            "configuration": self.config,
            "results": self.results,
            "performance_summary": self._calculate_performance_summary(),
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        output_path = f"{self.config.get('output_path', '../reports/')}validation_report.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Validation report saved to: {output_path}")
        return report
    
    def _calculate_performance_summary(self):
        """Calculate overall performance metrics."""
        summary = {}
        
        if 'stage_a' in self.results:
            stage_a = self.results['stage_a']
            summary['stage_a'] = {
                'accuracy': stage_a['accuracy'],
                'status': 'PASS' if stage_a['accuracy'] >= 0.8 else 'FAIL'
            }
        
        if 'stage_b' in self.results:
            stage_b = self.results['stage_b']
            summary['stage_b'] = {
                'accuracy': stage_b['accuracy'],
                'unknown_detection_rate': stage_b['unknown_detection_rate'],
                'status': 'PASS' if stage_b['accuracy'] >= 0.85 and stage_b['unknown_detection_rate'] <= 0.15 else 'FAIL'
            }
        
        return summary
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if 'stage_a' in self.results:
            if self.results['stage_a']['accuracy'] < 0.8:
                recommendations.append("Stage A accuracy below target. Consider retraining with more diverse data.")
        
        if 'stage_b' in self.results:
            stage_b = self.results['stage_b']
            if stage_b['accuracy'] < 0.85:
                recommendations.append("Stage B accuracy below target. Consider fine-tuning the model.")
            if stage_b['unknown_detection_rate'] > 0.15:
                recommendations.append("Unknown detection rate too high. Adjust thresholds or improve calibration.")
        
        if not recommendations:
            recommendations.append("Pipeline performance meets all targets. Ready for production deployment.")
        
        return recommendations

def main():
    """Run validation pipeline."""
    validator = MLPipelineValidator()
    
    # Note: This is a template - you'll need to provide actual test data
    print("Validation pipeline initialized.")
    print("To run validation, provide test data and call validator methods.")
    
    # Example usage:
    # test_data = [("path/to/image1.jpg", "Acacia_mearnsii"), ...]
    # validator.validate_stage_a(model, test_data)
    # validator.validate_stage_b(pipeline, test_data)
    # validator.generate_report()

if __name__ == "__main__":
    main()