#!/usr/bin/env python3
"""
PySpark-based Parquet Chunk Merger
Merges chunked parquet files created by the Amazon dataset pipeline
Memory-safe for large datasets (9M+ rows)
Enhanced with improved diagnostics and fault tolerance
Docker-compatible with Linux container paths
"""

import sys
import argparse
import shutil
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_spark_session(app_name: str = "ParquetChunkMerger", max_memory: str = "4g", partitions_count: int = 10) -> SparkSession:
    """Create optimized Spark session for parquet merging with enhanced fault tolerance"""
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", max_memory) \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.shuffle.partitions", str(partitions_count)) \
        .config("spark.network.timeout", "600s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()
    
    # Set log level to reduce verbose output
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Spark session created with driver memory: {max_memory}")
    logger.info(f"Spark shuffle partitions set to: {partitions_count}")
    logger.info(f"Spark network timeout: 600s, heartbeat interval: 60s")
    return spark

def merge_parquet_chunks(
    temp_dir: str, 
    output_path: str, 
    repartition_count: int = 10,
    spark_memory: str = "4g",
    skip_cleanup: bool = False
) -> dict:
    """
    Merge parquet chunk files using PySpark
    
    Args:
        temp_dir: Directory containing chunk*.parquet files
        output_path: Path for merged output file
        repartition_count: Number of partitions for output (controls memory usage)
        spark_memory: Maximum memory for Spark driver
        skip_cleanup: If True, keep temporary chunk files after merge
    
    Returns:
        dict: Results summary
    """
    
    temp_path = Path(temp_dir)
    output_file = Path(output_path)
    temp_output_path = Path(str(output_path) + "_temp")
    
    # Validate inputs
    if not temp_path.exists():
        raise FileNotFoundError(f"Temp directory not found: {temp_dir}")
    
    chunk_files = list(temp_path.glob("chunk_*.parquet"))
    if not chunk_files:
        raise ValueError(f"No chunk files found in {temp_dir}")
    
    logger.info(f"Found {len(chunk_files)} chunk files to merge")
    logger.info(f"Temp directory: {temp_dir}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Temporary output: {temp_output_path}")
    
    # Create Spark session with enhanced config
    spark = create_spark_session(max_memory=spark_memory, partitions_count=repartition_count)
    
    try:
        # Read all parquet chunks as a single DataFrame
        logger.info("Reading parquet chunks...")
        df = spark.read.parquet(str(temp_path / "chunk_*.parquet"))
        
        # Log DataFrame info
        total_rows = df.count()
        logger.info(f"Total rows across all chunks: {total_rows:,}")
        logger.info(f"Schema columns: {df.columns}")
        
        # Print and save schema for diagnostics
        logger.info("DataFrame schema:")
        df.printSchema()
        
        # Save schema to file for version control and debugging
        schema_file = output_file.parent / "schema.txt"
        try:
            with open(schema_file, 'w') as f:
                # Capture schema string representation
                schema_str = df._jdf.schema().treeString()
                f.write(f"Schema for {output_file.name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total columns: {len(df.columns)}\n")
                f.write(f"Total rows: {total_rows:,}\n\n")
                f.write("Column names:\n")
                for i, col_name in enumerate(df.columns, 1):
                    f.write(f"{i:2d}. {col_name}\n")
                f.write(f"\nDetailed schema:\n{schema_str}")
            logger.info(f"Schema saved to: {schema_file}")
        except Exception as e:
            logger.warning(f"Could not save schema file: {e}")
        
        # Use coalesce instead of repartition for better memory efficiency
        logger.info(f"Coalescing to {repartition_count} partitions for optimized write...")
        df_coalesced = df.coalesce(repartition_count)
        
        # Clean up any existing temp output directory
        if temp_output_path.exists():
            logger.info(f"Removing existing temporary output: {temp_output_path}")
            shutil.rmtree(temp_output_path)
        
        # Enhanced logging around the critical write operation
        logger.info("=" * 60)
        logger.info("STARTING PARQUET WRITE OPERATION")
        logger.info(f"Target location: {temp_output_path}")
        logger.info(f"Partitions: {repartition_count}")
        logger.info(f"Compression: snappy")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        # Write to temporary location first (safer write strategy)
        df_coalesced.write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(str(temp_output_path))
        
        logger.info("=" * 60)
        logger.info("PARQUET WRITE OPERATION COMPLETED")
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        # Verify output from temporary location
        logger.info("Verifying merged data...")
        merged_df = spark.read.parquet(str(temp_output_path))
        final_row_count = merged_df.count()
        
        # Validate row count match
        if final_row_count != total_rows:
            raise ValueError(f"Row count mismatch! Original: {total_rows:,}, Final: {final_row_count:,}")
        
        logger.info(f"Row count verification passed: {final_row_count:,}")
        
        # Verify partition files were created
        try:
            part_files = list(temp_output_path.glob("part-*.parquet"))
            logger.info(f"Partition verification: {len(part_files)} part files created")
            if len(part_files) == 0:
                logger.warning("No part-*.parquet files found - unexpected partitioning")
        except Exception as e:
            logger.warning(f"Could not verify partition files: {e}")
        
        # Move from temporary to final location (atomic-like operation)
        if output_file.exists():
            logger.info(f"Removing existing output file: {output_file}")
            if output_file.is_dir():
                shutil.rmtree(output_file)
            else:
                output_file.unlink()
        
        logger.info(f"Moving from temporary to final location: {output_file}")
        shutil.move(str(temp_output_path), str(output_file))
        
        # Safe file size logging
        file_size_mb = 0
        try:
            if output_file.is_dir():
                # For directory-based parquet, sum all part files
                total_size = sum(f.stat().st_size for f in output_file.rglob("*.parquet"))
                file_size_mb = total_size / 1024**2
            else:
                file_size_mb = output_file.stat().st_size / 1024**2
            logger.info(f"Output file size: {file_size_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"Could not determine output file size: {e}")
            file_size_mb = 0
        
        logger.info(f"Merge completed successfully!")
        logger.info(f"Final row count: {final_row_count:,}")
        
        # Cleanup temp files if merge successful and not skipped
        cleanup_performed = False
        if not skip_cleanup:
            if final_row_count == total_rows:
                logger.info("Cleaning up temporary chunk files...")
                try:
                    for chunk_file in chunk_files:
                        chunk_file.unlink()
                    temp_path.rmdir()
                    cleanup_performed = True
                    logger.info("Temporary files cleaned up")
                except Exception as e:
                    logger.warning(f"Could not clean up temporary files: {e}")
            else:
                logger.warning(f"Row count mismatch! Original: {total_rows}, Final: {final_row_count}")
                logger.warning("Keeping temporary files for debugging")
        else:
            logger.info("Cleanup skipped due to --skip-cleanup flag")
        
        return {
            'status': 'success',
            'input_chunks': len(chunk_files),
            'total_rows': final_row_count,
            'output_file': str(output_file),
            'file_size_mb': file_size_mb,
            'partitions_created': len(part_files) if 'part_files' in locals() else 0,
            'cleanup_performed': cleanup_performed,
            'schema_file': str(schema_file) if 'schema_file' in locals() else None
        }
        
    except Exception as e:
        logger.error(f"Merge failed: {str(e)}")
        # Clean up temp output on failure
        if temp_output_path.exists():
            try:
                shutil.rmtree(temp_output_path)
                logger.info("Cleaned up temporary output after failure")
            except Exception as cleanup_e:
                logger.warning(f"Could not clean up temporary output: {cleanup_e}")
        raise
    
    finally:
        # Stop Spark session
        spark.stop()
        logger.info("Spark session stopped")

def main():
    """Main execution with CLI argument parsing"""
    
    parser = argparse.ArgumentParser(description="Merge parquet chunk files using PySpark")
    
    parser.add_argument(
        "--temp-dir", 
        required=True, 
        help="Directory containing chunk*.parquet files"
    )
    
    parser.add_argument(
        "--output", 
        required=True,
        help="Output path for merged parquet file"
    )
    
    parser.add_argument(
        "--partitions", 
        type=int, 
        default=10,
        help="Number of partitions for repartitioning (default: 10)"
    )
    
    parser.add_argument(
        "--memory", 
        default="4g",
        help="Spark driver memory (default: 4g)"
    )
    
    parser.add_argument(
        "--skip-cleanup", 
        action="store_true",
        help="Keep temporary chunk files after merge (useful for debugging)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Execute merge
        result = merge_parquet_chunks(
            temp_dir=args.temp_dir,
            output_path=args.output,
            repartition_count=args.partitions,
            spark_memory=args.memory,
            skip_cleanup=args.skip_cleanup
        )
        
        # Print enhanced summary
        print(f"\nMERGE COMPLETED SUCCESSFULLY!")
        print(f"Input chunks: {result['input_chunks']}")
        print(f"Total rows: {result['total_rows']:,}")
        print(f"Output file: {result['output_file']}")
        print(f"File size: {result['file_size_mb']:.1f} MB")
        print(f"Partitions created: {result['partitions_created']}")
        if result['schema_file']:
            print(f"Schema saved: {result['schema_file']}")
        if result['cleanup_performed']:
            print(f"Temporary files cleaned up")
        else:
            print(f"Temporary files preserved")
        print(f"\nNext: Update your pipeline config and restart the main script")
        
    except Exception as e:
        print(f"\nMERGE FAILED: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example usage for Docker container:
# python merge_parquet_chunks.py --temp-dir "/mnt/amazon_data/processed/temp_Sports_and_Outdoors" --output "/mnt/amazon_data/processed/9m_reviews.parquet"
# python merge_parquet_chunks.py --temp-dir "/mnt/amazon_data/processed/temp_meta_Sports_and_Outdoors" --output "/mnt/amazon_data/processed/metadata_full.parquet" --partitions 5 --memory 6g
# python merge_parquet_chunks.py --temp-dir "/mnt/amazon_data/processed/temp_chunks" --output "/mnt/amazon_data/processed/merged_data.parquet" --skip-cleanup --verbose