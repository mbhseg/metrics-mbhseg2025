#!/usr/bin/env python3
"""
Competition Evaluation Pipeline
Automated competition evaluation system: batch evaluate prediction results, generate comprehensive ranking reports
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback
from tqdm import tqdm

# Import evaluation functions
from diverse_performance import evaluate_diverse_performance
from personalized_performance import evaluate_personalized_performance
from auto_config import get_auto_config


class CompetitionEvaluator:
    """Competition Evaluator: Automated batch evaluation pipeline"""

    def __init__(
        self,
        predictions_path: str,
        ground_truth_path: str,
        output_dir: str = "competition_results",
    ):
        """
        Initialize competition evaluator

        Args:
            predictions_path: Predictions folder path
            ground_truth_path: Ground truth folder path
            output_dir: Results output directory
        """
        self.predictions_path = Path(predictions_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Result storage
        self.diverse_results = []
        self.personalized_results = []
        self.failed_samples = []
        self.sample_mapping = {}

    def discover_samples(self) -> List[Tuple[str, str, str]]:
        """
        Discover and match prediction files with ground truth folders

        Returns:
            List of (sample_id, pred_file_path, gt_folder_path) tuples
        """
        print("🔍 Discovering prediction files and matching ground truth...")

        # Get all prediction files
        pred_files = list(self.predictions_path.glob("*.nii.gz"))
        pred_files.sort()

        print(f"📁 Found {len(pred_files)} prediction files")

        sample_pairs = []
        missing_gt = []

        for pred_file in pred_files:
            # Extract sample ID from prediction file name
            # Note: pred_file.stem only removes the last .gz, so we need to handle it
            sample_id = pred_file.name.replace(".nii.gz", "")

            # Construct corresponding GT folder path
            gt_folder = self.ground_truth_path / sample_id

            if gt_folder.exists() and gt_folder.is_dir():
                # Check if necessary annotation files exist
                label_files = list(gt_folder.glob("label_annot_*.nii.gz"))
                if len(label_files) >= 2:  # Need at least 2 expert annotations
                    sample_pairs.append((sample_id, str(pred_file), str(gt_folder)))
                    self.sample_mapping[sample_id] = {
                        "pred_file": str(pred_file),
                        "gt_folder": str(gt_folder),
                        "gt_files": [str(f) for f in label_files],
                    }
                else:
                    missing_gt.append(
                        f"{sample_id} (missing expert annotation files, only has {len(label_files)})"
                    )
            else:
                missing_gt.append(f"{sample_id} (GT folder does not exist)")

        if missing_gt:
            print(
                f"❌ Found {len(missing_gt)} samples missing corresponding ground truth:"
            )
            for missing in missing_gt[:10]:  # Only show first 10
                print(f"   - {missing}")
            if len(missing_gt) > 10:
                print(f"   ... and {len(missing_gt) - 10} more")
            raise FileNotFoundError(
                f"Cannot find ground truth for {len(missing_gt)} samples. Please check data integrity."
            )

        print(f"✅ Successfully matched {len(sample_pairs)} samples")
        return sample_pairs

    def evaluate_single_sample(
        self, sample_id: str, pred_file: str, gt_folder: str
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Evaluate single sample

        Args:
            sample_id: Sample ID
            pred_file: Prediction file path
            gt_folder: GT folder path

        Returns:
            (diverse_result, personalized_result) or (None, None) if failed
        """
        try:
            # Use auto_config to get parameters
            auto_config, _ = get_auto_config(pred_file, gt_folder, verbose=False)

            # For single file evaluation, create temporary directory with just this file
            import tempfile
            import shutil
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy single prediction file to temp directory
                temp_pred_file = os.path.join(temp_dir, os.path.basename(pred_file))
                shutil.copy2(pred_file, temp_pred_file)
                
                # Diverse Performance evaluation using temp directory
                diverse_result = evaluate_diverse_performance(
                    temp_dir,
                    gt_folder,
                    None,  # Pass directory path, output=None means don't save files
                    pred_pattern=auto_config["pred_pattern"],
                    gt_pattern=auto_config["gt_pattern"],
                    multiclass=auto_config["multiclass"],
                    num_classes=auto_config["num_classes"],
                    exclude_background=auto_config["exclude_background"],
                )

                # Personalized Performance evaluation using same temp directory
                personalized_result = evaluate_personalized_performance(
                    temp_dir,
                    gt_folder,
                    None,  # Pass directory path, output=None means don't save files
                    pred_pattern=auto_config["pred_pattern"],
                    gt_pattern=auto_config["gt_pattern"],
                    multiclass=auto_config["multiclass"],
                    num_classes=auto_config["num_classes"],
                    exclude_background=auto_config["exclude_background"],
                )

            return diverse_result, personalized_result

        except Exception as e:
            print(f"❌ Sample {sample_id} evaluation failed: {str(e)}")
            self.failed_samples.append(
                {
                    "sample_id": sample_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
            return None, None

    def run_batch_evaluation(self, sample_pairs: List[Tuple[str, str, str]]) -> None:
        """
        Run batch evaluation

        Args:
            sample_pairs: (sample_id, pred_file, gt_folder) tuples
        """
        print(f"🚀 Starting batch evaluation of {len(sample_pairs)} samples...")

        success_count = 0

        # Use tqdm to show progress bar
        for sample_id, pred_file, gt_folder in tqdm(
            sample_pairs, desc="Evaluation Progress"
        ):
            diverse_result, personalized_result = self.evaluate_single_sample(
                sample_id, pred_file, gt_folder
            )

            if diverse_result is not None and personalized_result is not None:
                # Add sample ID to results
                diverse_result["sample_id"] = sample_id
                personalized_result["sample_id"] = sample_id

                self.diverse_results.append(diverse_result)
                self.personalized_results.append(personalized_result)
                success_count += 1

        print("\n📊 Batch evaluation completed:")
        print(f"   ✅ Successfully evaluated: {success_count} samples")
        print(f"   ❌ Failed samples: {len(self.failed_samples)} samples")

    def compute_aggregate_metrics(self) -> Dict:
        """
        Calculate aggregate metrics (means and standard deviations)

        Returns:
            Dictionary containing all aggregate metrics
        """
        print("📈 Computing aggregate metrics...")

        if not self.diverse_results or not self.personalized_results:
            raise ValueError(
                "No successful evaluation results available, cannot compute aggregate metrics"
            )

        aggregate_results = {
            "num_samples": len(self.diverse_results),
            "num_failed": len(self.failed_samples),
            "diverse_performance": {},
            "personalized_performance": {},
        }

        # === Diverse Performance Aggregation ===
        # User explicitly requested 4 metrics: GED, Dice_soft, Dice_match, Dice_max
        diverse_metrics = ["GED", "Dice_soft", "Dice_match", "Dice_max"]

        for metric in diverse_metrics:
            values = [
                result[metric] for result in self.diverse_results if metric in result
            ]
            if values:
                # Use nanmean, nanstd etc. to handle NaN values
                values_array = np.array(values)
                aggregate_results["diverse_performance"][f"{metric}_mean"] = np.nanmean(
                    values_array
                )
                aggregate_results["diverse_performance"][f"{metric}_std"] = np.nanstd(
                    values_array
                )
                aggregate_results["diverse_performance"][f"{metric}_min"] = np.nanmin(
                    values_array
                )
                aggregate_results["diverse_performance"][f"{metric}_max"] = np.nanmax(
                    values_array
                )

        # === Personalized Performance Aggregation ===
        # User requirement: Dice score for each expert, then take mean as final metric
        dice_each_mean_values = [
            result["Dice_each_mean"]
            for result in self.personalized_results
            if "Dice_each_mean" in result
        ]

        if dice_each_mean_values:
            aggregate_results["personalized_performance"]["Dice_each_mean_mean"] = (
                np.mean(dice_each_mean_values)
            )
            aggregate_results["personalized_performance"]["Dice_each_mean_std"] = (
                np.std(dice_each_mean_values)
            )
            aggregate_results["personalized_performance"]["Dice_each_mean_min"] = (
                np.min(dice_each_mean_values)
            )
            aggregate_results["personalized_performance"]["Dice_each_mean_max"] = (
                np.max(dice_each_mean_values)
            )

        # Also provide aggregate statistics for individual experts (optional)
        for result in self.personalized_results:
            for key in result:
                if key.startswith("Dice_expert_"):
                    expert_values = [
                        r[key] for r in self.personalized_results if key in r
                    ]
                    if expert_values:
                        aggregate_results["personalized_performance"][f"{key}_mean"] = (
                            np.mean(expert_values)
                        )
                        aggregate_results["personalized_performance"][f"{key}_std"] = (
                            np.std(expert_values)
                        )

        return aggregate_results

    def save_results(self, aggregate_results: Dict) -> None:
        """
        Save evaluation results

        Args:
            aggregate_results: Aggregate results dictionary
        """
        print("💾 Saving evaluation results...")

        # Convert numpy types to Python native types, to solve JSON serialization issues
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj

        aggregate_results_clean = convert_numpy_types(aggregate_results)

        # Save aggregate results
        aggregate_file = self.output_dir / "competition_aggregate_results.json"
        with open(aggregate_file, "w", encoding="utf-8") as f:
            json.dump(aggregate_results_clean, f, indent=2, ensure_ascii=False)

        # Save detailed results
        detailed_results = {
            "sample_mapping": self.sample_mapping,
            "diverse_results": self.diverse_results,
            "personalized_results": self.personalized_results,
            "failed_samples": self.failed_samples,
        }

        # Also convert numpy types for detailed results
        detailed_results_clean = convert_numpy_types(detailed_results)

        detailed_file = self.output_dir / "competition_detailed_results.json"
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(detailed_results_clean, f, indent=2, ensure_ascii=False)

        # Generate human-readable report
        self.generate_report(aggregate_results)

        print("📄 Results saved to:")
        print(f"   - {aggregate_file}")
        print(f"   - {detailed_file}")
        print(f"   - {self.output_dir}/competition_report.txt")

    def generate_report(self, aggregate_results: Dict) -> None:
        """
        Generate human-readable competition report

        Args:
            aggregate_results: Aggregate results dictionary
        """
        report_file = self.output_dir / "competition_report.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("COMPETITION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Basic statistics
            f.write("📊 Evaluation Overview\n")
            f.write(f"Total samples: {aggregate_results['num_samples']}\n")
            f.write(f"Failed samples: {aggregate_results['num_failed']}\n")
            f.write(
                f"Success rate: {aggregate_results['num_samples'] / (aggregate_results['num_samples'] + aggregate_results['num_failed']) * 100:.2f}%\n\n"
            )

            # Diverse Performance results
            f.write("🎯 DIVERSE PERFORMANCE METRICS\n")
            f.write("-" * 50 + "\n")
            diverse = aggregate_results["diverse_performance"]

            for metric in ["GED", "Dice_soft", "Dice_match", "Dice_max"]:
                if f"{metric}_mean" in diverse:
                    mean_val = diverse[f"{metric}_mean"]
                    std_val = diverse[f"{metric}_std"]
                    min_val = diverse[f"{metric}_min"]
                    max_val = diverse[f"{metric}_max"]
                    f.write(
                        f"{metric:<12}: {mean_val:.4f} ± {std_val:.4f} [{min_val:.4f}, {max_val:.4f}]\n"
                    )

            f.write("\n")

            # Personalized Performance results
            f.write("👥 PERSONALIZED PERFORMANCE METRICS\n")
            f.write("-" * 50 + "\n")
            personalized = aggregate_results["personalized_performance"]

            if "Dice_each_mean_mean" in personalized:
                mean_val = personalized["Dice_each_mean_mean"]
                std_val = personalized["Dice_each_mean_std"]
                min_val = personalized["Dice_each_mean_min"]
                max_val = personalized["Dice_each_mean_max"]
                f.write(
                    f"{'Overall':<12}: {mean_val:.4f} ± {std_val:.4f} [{min_val:.4f}, {max_val:.4f}]\n"
                )

            # Statistics for individual experts
            for key in personalized:
                if key.startswith("Dice_expert_") and key.endswith("_mean"):
                    expert_id = key.split("_")[2]
                    mean_val = personalized[key]
                    std_key = key.replace("_mean", "_std")
                    std_val = personalized.get(std_key, 0.0)
                    f.write(
                        f"{'Expert_' + expert_id:<12}: {mean_val:.4f} ± {std_val:.4f}\n"
                    )

            # Failed samples report
            if self.failed_samples:
                f.write(f"\n❌ FAILED SAMPLES ({len(self.failed_samples)})\n")
                f.write("-" * 50 + "\n")
                for i, failed in enumerate(
                    self.failed_samples[:10]
                ):  # Only show first 10
                    f.write(f"{i + 1}. {failed['sample_id']}: {failed['error']}\n")
                if len(self.failed_samples) > 10:
                    f.write(
                        f"... and {len(self.failed_samples) - 10} more failed samples\n"
                    )

    def run_competition_evaluation(self) -> Dict:
        """
        Run complete competition evaluation pipeline

        Returns:
            Aggregate evaluation results
        """
        print("🏆 Starting competition evaluation pipeline...")

        # 1. Discover samples
        sample_pairs = self.discover_samples()

        # 2. Batch evaluation
        self.run_batch_evaluation(sample_pairs)

        # 3. Calculate aggregate metrics
        aggregate_results = self.compute_aggregate_metrics()

        # 4. Save results
        self.save_results(aggregate_results)

        print("🎉 Competition evaluation completed!")
        return aggregate_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Competition Evaluation Pipeline")
    parser.add_argument("--pred_path", required=True, help="Path to predictions folder")
    parser.add_argument(
        "--gt_path",
        required=True,
        help="Path to ground truth folder (MBH_val_label_2025)",
    )
    parser.add_argument(
        "--output_dir",
        default="competition_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    try:
        evaluator = CompetitionEvaluator(
            predictions_path=args.pred_path,
            ground_truth_path=args.gt_path,
            output_dir=args.output_dir,
        )

        aggregate_results = evaluator.run_competition_evaluation()

        # Print key results
        print("\n🏆 Competition Final Results:")
        diverse = aggregate_results["diverse_performance"]
        personalized = aggregate_results["personalized_performance"]

        print("Diverse Performance:")
        for metric in ["GED", "Dice_soft", "Dice_match", "Dice_max"]:
            if f"{metric}_mean" in diverse:
                print(
                    f"  {metric}: {diverse[f'{metric}_mean']:.4f} ± {diverse[f'{metric}_std']:.4f}"
                )

        print("Personalized Performance:")
        if "Dice_each_mean_mean" in personalized:
            print(
                f"  Overall: {personalized['Dice_each_mean_mean']:.4f} ± {personalized['Dice_each_mean_std']:.4f}"
            )

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
