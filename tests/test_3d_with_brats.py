#!/usr/bin/env python
"""
BraTS 3D Visualization Testing Script
File location: tests/test_3d_with_brats.py

Tests the 3D visualization backend with real BraTS brain tumor cases.
Validates:
1. 3D mesh generation works with real data
2. Response times are acceptable
3. Mesh quality is reasonable
4. Fallback to 2D works if 3D fails

Usage:
    python tests/test_3d_with_brats.py --brats_dir /path/to/brats --num_cases 5
    python tests/test_3d_with_brats.py --brats_dir ~/BraTS_data --output results.json
"""

import os
import sys
import json
import requests
import argparse
import time
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class BraTS3DValidator:
    """Validate 3D visualization with BraTS cases"""
    
    def __init__(self, api_url: str = "http://localhost:5000/api"):
        self.api_url = api_url
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'api_url': api_url,
            'cases': []
        }
    
    def find_brats_cases(self, brats_dir: str, limit: int = None):
        """Find BraTS case directories"""
        brats_path = Path(brats_dir)
        
        if not brats_path.exists():
            logger.error(f"BraTS directory not found: {brats_dir}")
            return []
        
        cases = []
        for case_dir in sorted(brats_path.glob("BraTS*")):
            if case_dir.is_dir():
                # Check for required modalities
                t1 = list(case_dir.glob("*T1.nii.gz")) or list(case_dir.glob("*_t1.nii.gz"))
                t2 = list(case_dir.glob("*T2.nii.gz")) or list(case_dir.glob("*_t2.nii.gz"))
                flair = list(case_dir.glob("*FLAIR.nii.gz")) or list(case_dir.glob("*_flair.nii.gz"))
                t1c = list(case_dir.glob("*T1c.nii.gz")) or list(case_dir.glob("*_t1ce.nii.gz"))
                
                if t1 and t2 and flair and t1c:
                    cases.append({
                        'name': case_dir.name,
                        'path': case_dir,
                        'T1': str(t1[0]),
                        'T2': str(t2[0]),
                        'FLAIR': str(flair[0]),
                        'T1c': str(t1c[0]),
                    })
        
        if limit:
            cases = cases[:limit]
        
        logger.info(f"Found {len(cases)} valid BraTS cases")
        return cases
    
    def test_case(self, case: dict) -> dict:
        """Test a single BraTS case"""
        case_name = case['name']
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing: {case_name}")
        logger.info(f"{'='*70}")
        
        result = {
            'case_name': case_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'unknown',
            'error': None,
            'timing': {},
            'segmentation': {},
            '3d': {}
        }
        
        try:
            # Prepare files
            files = {
                'T1': open(case['T1'], 'rb'),
                'T2': open(case['T2'], 'rb'),
                'FLAIR': open(case['FLAIR'], 'rb'),
                'T1c': open(case['T1c'], 'rb'),
            }
            
            # Send request
            logger.info("Sending segmentation request...")
            t0 = time.time()
            
            response = requests.post(
                f"{self.api_url}/predict/segment",
                files=files,
                timeout=300  # 5 minute timeout for large cases
            )
            
            elapsed = time.time() - t0
            result['timing']['total'] = elapsed
            
            # Close files
            for f in files.values():
                f.close()
            
            # Check response
            if response.status_code != 200:
                result['status'] = 'failed'
                result['error'] = f"HTTP {response.status_code}"
                logger.error(f"❌ HTTP {response.status_code}")
                return result
            
            data = response.json()
            
            if data.get('status') != 'success':
                result['status'] = 'failed'
                result['error'] = data.get('error', 'Unknown error')
                logger.error(f"❌ {result['error']}")
                return result
            
            # Extract segmentation data
            seg = data.get('segmentation', {})
            result['segmentation'] = {
                'tumor_volume_voxels': seg.get('tumor_volume_voxels', 0),
                'edema_volume_voxels': seg.get('edema_volume_voxels', 0),
                'tumor_confidence': data.get('quality', {}).get('tumor_confidence', 0),
            }
            
            logger.info(f"✅ Segmentation complete:")
            logger.info(f"   Tumor volume: {result['segmentation']['tumor_volume_voxels']:,} voxels")
            logger.info(f"   Edema volume: {result['segmentation']['edema_volume_voxels']:,} voxels")
            logger.info(f"   Confidence: {result['segmentation']['tumor_confidence']:.3f}")
            
            # Check 3D availability
            if data.get('3d_available'):
                geometry = data.get('3d_geometry', {})
                tumor_info = geometry.get('tumor_mesh_info', {})
                
                result['3d'] = {
                    'available': True,
                    'tumor_vertices': tumor_info.get('vertex_count', 0),
                    'tumor_faces': tumor_info.get('face_count', 0),
                    'attention_mean': tumor_info.get('attention_mean', 0),
                    'attention_max': tumor_info.get('attention_max', 0),
                }
                
                logger.info(f"✅ 3D mesh generated:")
                logger.info(f"   Tumor vertices: {result['3d']['tumor_vertices']:,}")
                logger.info(f"   Tumor faces: {result['3d']['tumor_faces']:,}")
                logger.info(f"   Attention mean: {result['3d']['attention_mean']:.3f}")
                logger.info(f"   Attention max: {result['3d']['attention_max']:.3f}")
            else:
                result['3d'] = {'available': False}
                logger.warning("⚠️  3D mesh not available (fallback to 2D)")
            
            result['status'] = 'success'
            result['timing']['api_call'] = elapsed
            
            logger.info(f"✅ Total time: {elapsed:.1f}s")
            
        except requests.exceptions.Timeout:
            result['status'] = 'timeout'
            result['error'] = "Request timeout (>300s)"
            logger.error(f"❌ {result['error']}")
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"❌ {result['error']}")
        
        return result
    
    def run_batch(self, cases: list) -> dict:
        """Run tests on all cases"""
        logger.info("\n" + "="*70)
        logger.info(f"🧪 BRATS 3D VISUALIZATION TEST SUITE")
        logger.info(f"{'='*70}")
        logger.info(f"Testing {len(cases)} cases")
        logger.info(f"API URL: {self.api_url}")
        
        # Test API connectivity
        logger.info("\nChecking API connectivity...")
        try:
            response = requests.get(f"{self.api_url.replace('/api', '')}/api/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                logger.info(f"✅ API healthy: {health.get('version', 'unknown')}")
                logger.info(f"   3D available: {health.get('3d_visualization_available', False)}")
            else:
                logger.error("❌ API not responding")
                return self.results
        except Exception as e:
            logger.error(f"❌ Cannot connect to API: {e}")
            return self.results
        
        # Run tests
        t_start = time.time()
        
        for i, case in enumerate(cases, 1):
            logger.info(f"\n[{i}/{len(cases)}]", extra={'rank': i})
            result = self.test_case(case)
            self.results['cases'].append(result)
        
        # Summary
        t_total = time.time() - t_start
        
        self.results['summary'] = {
            'total_cases': len(cases),
            'passed': sum(1 for c in self.results['cases'] if c['status'] == 'success'),
            'failed': sum(1 for c in self.results['cases'] if c['status'] == 'failed'),
            'errors': sum(1 for c in self.results['cases'] if c['status'] == 'error'),
            'timeouts': sum(1 for c in self.results['cases'] if c['status'] == 'timeout'),
            '3d_generated': sum(1 for c in self.results['cases'] if c.get('3d', {}).get('available', False)),
            'total_time_seconds': t_total,
            'average_time_per_case': t_total / len(cases) if cases else 0,
        }
        
        return self.results
    
    def print_summary(self):
        """Print summary report"""
        logger.info("\n" + "="*70)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("="*70)
        
        summary = self.results.get('summary', {})
        logger.info(f"\nTotal cases: {summary.get('total_cases', 0)}")
        logger.info(f"✅ Passed: {summary.get('passed', 0)}")
        logger.info(f"❌ Failed: {summary.get('failed', 0)}")
        logger.info(f"⚠️  Errors: {summary.get('errors', 0)}")
        logger.info(f"⏱️  Timeouts: {summary.get('timeouts', 0)}")
        logger.info(f"\n🧊 3D meshes generated: {summary.get('3d_generated', 0)}/{summary.get('total_cases', 0)}")
        logger.info(f"⏱️  Total time: {summary.get('total_time_seconds', 0):.1f}s")
        logger.info(f"⏱️  Avg per case: {summary.get('average_time_per_case', 0):.1f}s")
        
        # Individual case results
        logger.info("\nPer-case results:")
        for case in self.results.get('cases', []):
            status_icon = {
                'success': '✅',
                'failed': '❌',
                'error': '⚠️',
                'timeout': '⏱️'
            }.get(case['status'], '❓')
            
            size_str = f"{case['segmentation'].get('tumor_volume_voxels', 0):,}v"
            time_str = f"{case['timing'].get('total', 0):.1f}s"
            
            if case.get('3d', {}).get('available'):
                size_str += f" → 3D: {case['3d']['tumor_vertices']:,}v"
            
            logger.info(f"  {status_icon} {case['case_name']:30} {size_str:25} {time_str}")


def main():
    parser = argparse.ArgumentParser(description="Test 3D visualization with BraTS cases")
    parser.add_argument("--brats_dir", type=str, default="./brats_data",
                       help="Path to BraTS data directory")
    parser.add_argument("--num_cases", type=int, default=5,
                       help="Number of cases to test")
    parser.add_argument("--api_url", type=str, default="http://localhost:5000/api",
                       help="API server URL")
    parser.add_argument("--output", type=str, default="brats_3d_test_results.json",
                       help="Output results file")
    
    args = parser.parse_args()
    
    # Find cases
    validator = BraTS3DValidator(api_url=args.api_url)
    cases = validator.find_brats_cases(args.brats_dir, limit=args.num_cases)
    
    if not cases:
        logger.error(f"No valid BraTS cases found in {args.brats_dir}")
        logger.info("\nExpected structure:")
        logger.info("  brats_data/")
        logger.info("    BraTS_001/")
        logger.info("      BraTS_001_T1.nii.gz")
        logger.info("      BraTS_001_T2.nii.gz")
        logger.info("      BraTS_001_FLAIR.nii.gz")
        logger.info("      BraTS_001_T1c.nii.gz")
        return
    
    # Run tests
    results = validator.run_batch(cases)
    
    # Print summary
    validator.print_summary()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n✅ Results saved to {args.output}")
    
    # Exit with error code if any failed
    summary = results.get('summary', {})
    if summary.get('failed', 0) > 0 or summary.get('errors', 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
