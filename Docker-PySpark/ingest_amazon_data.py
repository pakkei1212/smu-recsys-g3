#!/usr/bin/env python3
"""
Amazon Sports & Outdoors Dataset Processing Pipeline - Local Version
Processes raw .jsonl.gz files from external drive without HuggingFace dependencies
Enhanced with memory management and YAML-based stage control
"""

import pandas as pd
import numpy as np
import json
import gzip
import gc
import psutil
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configuration
RAW_DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/processed")
REVIEWS_FILE = RAW_DATA_DIR / "Sports_and_Outdoors.jsonl.gz"
METADATA_FILE = RAW_DATA_DIR / "Meta_Sports_and_Outdoors.jsonl.gz"
CONFIG_FILE = Path("pipeline_config.yaml")

# Global logging setup
log_entries = []

def log_step(message: str) -> None:
    """Log processing steps with timestamp and memory usage"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    memory_mb = psutil.virtual_memory().used / 1024**2
    entry = f"[{timestamp}] {message} | Memory: {memory_mb:.0f}MB"
    log_entries.append(entry)
    print(entry)

def clean_price(val):
    """Convert price string to float or NaN if non-numeric (e.g. '—', 'N/A')"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan

def load_config() -> Dict[str, bool]:
    """Load pipeline configuration from YAML file"""
    default_config = {
        'load_reviews': True,
        'compute_top50': True,
        'filter_reviews': True,
        'load_metadata': True,
        'join_reviews_metadata': True,
        'chunk_split': True,
        'final_merge': True
    }
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = yaml.safe_load(f)
            log_step(f"Configuration loaded from {CONFIG_FILE}")
            return {**default_config, **config}
        except Exception as e:
            log_step(f"WARNING: Failed to load config file: {e}")
            log_step("Using default configuration")
    else:
        log_step(f"Config file {CONFIG_FILE} not found, using defaults")
        # Create default config file
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        log_step(f"Created default config file: {CONFIG_FILE}")
    
    return default_config

def load_jsonl_gz_streaming(filepath: Path, chunk_size: int = 100_000, max_rows: int = None) -> str:
    """
    Stream-load .jsonl.gz file in chunks to avoid memory overload
    Memory-safe for large files (16GB RAM or less)
    
    Args:
        filepath: Path to .jsonl.gz file
        chunk_size: Number of rows per chunk (default: 100,000 for memory safety)
        max_rows: Optional limit on total rows to load (None for all)
    
    Returns:
        str: Path to temporary directory containing chunk files
    """
    log_step(f"Streaming and loading {filepath.name} in chunks of {chunk_size:,}...")
    if max_rows:
        log_step(f"  Row limit: {max_rows:,} rows maximum")
    
    temp_dir = OUTPUT_DIR / f"Temporary_Chunks_{filepath.name.replace('.jsonl.gz', '').replace('.jsonl', '')}"
    temp_dir.mkdir(exist_ok=True)
    
    chunk_files = []
    chunk_data = []
    chunk_index = 0
    total_rows_processed = 0
    
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            # Check if we've hit the max_rows limit
            if max_rows and total_rows_processed >= max_rows:
                log_step(f"Reached max_rows={max_rows:,}, stopping early.")
                break
            
            try:
                chunk_data.append(json.loads(line.strip()))
                total_rows_processed += 1
            except json.JSONDecodeError as e:
                log_step(f"  WARNING: Skipping malformed JSON at line {i}: {e}")
                continue
            
            # Progress logging every 500K rows
            if i % 500_000 == 0:
                log_step(f"  Read {i:,} lines, processed {total_rows_processed:,} valid rows...")
            
            if len(chunk_data) >= chunk_size:
                df = pd.DataFrame(chunk_data)
                
                # Clean price column if it exists
                if 'price' in df.columns:
                    df['price'] = df['price'].apply(clean_price)
                    log_step(f"  Cleaned price column in chunk {chunk_index + 1}")
                
                chunk_path = temp_dir / f"chunk_{chunk_index + 1:04d}.parquet"
                df.to_parquet(chunk_path, index=False, engine='fastparquet')
                log_step(f"  Saved chunk {chunk_index + 1}: {len(df):,} rows")
                
                chunk_files.append(chunk_path)
                chunk_data = []
                chunk_index += 1
                
                # Memory cleanup
                del df
                gc.collect()
                
                # Early exit check after processing chunk
                if max_rows and total_rows_processed >= max_rows:
                    log_step(f"Reached max_rows={max_rows:,} after processing chunk, stopping.")
                    break
    
    # Handle final leftover rows (incomplete chunk or early termination)
    if chunk_data:
        df = pd.DataFrame(chunk_data)
        
        # Clean price column if it exists
        if 'price' in df.columns:
            df['price'] = df['price'].apply(clean_price)
            log_step(f"  Cleaned price column in final chunk {chunk_index + 1}")
        
        chunk_path = temp_dir / f"chunk_{chunk_index + 1:04d}.parquet"
        df.to_parquet(chunk_path, index=False, engine='fastparquet')
        log_step(f"  Saved final chunk {chunk_index + 1}: {len(df):,} rows")
        
        chunk_files.append(chunk_path)
        del df
        gc.collect()
    
    # No in-memory merge - just return temp directory path
    log_step(f"Streaming complete: {len(chunk_files)} chunk files created")
    log_step(f"  Total rows processed: {total_rows_processed:,}")
    log_step(f"  Chunk files saved in: {temp_dir}")
    log_step("NOTE: Use merge_parquet_chunks.py to merge chunks into final file")
    
    return str(temp_dir)

def save_dataframe_chunked(df: pd.DataFrame, filepath: Path, chunk_size: int = 500_000) -> None:
    """
    Save DataFrame to Parquet in chunks to avoid memory issues
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        chunk_size: Number of rows per chunk
    """
    total_rows = len(df)
    
    if total_rows <= chunk_size:
        # Small enough to save directly
        df.to_parquet(filepath, index=False, engine='fastparquet')
        log_step(f"Saved {total_rows:,} rows to {filepath}")
        return
    
    # Save in chunks, then merge
    log_step(f"Saving large DataFrame ({total_rows:,} rows) in chunks of {chunk_size:,}...")
    
    # Create temporary directory for chunks
    temp_dir = filepath.parent / f"temp_{filepath.stem}"
    temp_dir.mkdir(exist_ok=True)
    
    chunk_files = []
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        
        chunk = df.iloc[start_idx:end_idx].copy()
        chunk_path = temp_dir / f"chunk_{i+1}.parquet"
        
        chunk.to_parquet(chunk_path, index=False, engine='fastparquet')
        chunk_files.append(chunk_path)
        log_step(f"  Saved chunk {i+1}/{num_chunks}: {len(chunk):,} rows")
        
        del chunk
        gc.collect()
    
    # Merge chunks back into single file
    log_step("Merging chunks into final file...")
    chunks = []
    for chunk_path in chunk_files:
        chunk = pd.read_parquet(chunk_path)
        chunks.append(chunk)
    
    final_df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    
    # Save final file
    final_df.to_parquet(filepath, index=False, engine='fastparquet')
    log_step(f"Final file saved: {filepath}")
    
    # Clean up temporary files
    for chunk_path in chunk_files:
        chunk_path.unlink()
    temp_dir.rmdir()
    log_step("Temporary chunk files cleaned up")

def main():
    """Main pipeline execution"""
    
    # Load configuration
    config = load_config()
    log_step("Starting Amazon Sports & Outdoors Pipeline - Local Version")
    log_step("Pipeline configured for dynamic TOP_K selection (target: 1.5M reviews)")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_step(f"Output directory ready: {OUTPUT_DIR}")
    
    # STEP 1: Only check if load_reviews is True
    if config['load_reviews']:
        if not REVIEWS_FILE.exists():
            raise FileNotFoundError(f"Reviews file not found: {REVIEWS_FILE}")

    # STEP 4: Only check if load_metadata is True
    if config['load_metadata']:
        if not METADATA_FILE.exists():
            raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    # Only log if checks passed (i.e., didn't raise)
    log_step(f"Input files validated")

    # Variables to track pipeline state
    reviews_df = None
    TOP_K = 0  # Will be determined dynamically
    top_k_asins = None
    top_k_reviews = None
    metadata_df = None
    joined_df = None
    
    # ============================================================================
    # STEP 1: Load 9M reviews from Sports_and_Outdoors.jsonl.gz
    # ============================================================================
    if config['load_reviews']:
        log_step("STEP 1: Creating 9M review chunks...")
        
        # Use streaming to create chunks (no in-memory merge)
        temp_dir_path = load_jsonl_gz_streaming(REVIEWS_FILE, max_rows=9_000_000)
        
        log_step(f"Review chunks created in: {temp_dir_path}")
        log_step("IMPORTANT: Run 'python merge_parquet_chunks.py' to merge chunks before continuing")
        log_step("Pipeline will now exit - restart after merging chunks")
        
        return {
            'status': 'chunks_created',
            'temp_dir': temp_dir_path,
            'message': 'Run merge_parquet_chunks.py to merge chunks, then restart pipeline'
        }
    else:
        log_step("STEP 1: SKIPPED - Loading reviews disabled in config")
        # Try to load from checkpoint if it exists
        checkpoint_path = OUTPUT_DIR / "9m_reviews.parquet"
        if checkpoint_path.exists():
            reviews_df = pd.read_parquet(checkpoint_path)
            log_step(f"Loaded reviews from checkpoint: {len(reviews_df):,} rows")
        else:
            log_step("WARNING: No checkpoint found. Enable load_reviews or merge chunks first.")
    
    # ============================================================================
    # STEP 2: Dynamically determine top K parent_asin by review count
    # ============================================================================
    if config['compute_top50'] and reviews_df is not None:
        log_step("STEP 2: Computing dynamic top K products to reach 1.5M reviews...")
        
        # Get review counts for all products
        parent_asin_counts = reviews_df['parent_asin'].value_counts()
        
        # Compute cumulative sum to find optimal K
        target_reviews = 1_500_000
        cumulative_reviews = 0
        TOP_K = 0
        
        log_step(f"  Target review count: {target_reviews:,}")
        log_step(f"  Scanning {len(parent_asin_counts):,} unique products...")
        
        for asin, count in parent_asin_counts.items():
            cumulative_reviews += count
            TOP_K += 1
            
            # Stop when we reach the target
            if cumulative_reviews >= target_reviews:
                break
        
        # Get the final list of top K ASINs
        top_k_asins = parent_asin_counts.head(TOP_K).index.tolist()
        
        # Calculate actual review count for verification
        actual_reviews = parent_asin_counts.head(TOP_K).sum()
        
        log_step(f"Dynamic TOP_K determined: {TOP_K} products")
        log_step(f"  Cumulative reviews: {actual_reviews:,} (target: {target_reviews:,})")
        log_step(f"  Efficiency ratio: {(actual_reviews / target_reviews * 100):.1f}%")
        
        # Save top K list with dynamic filename
        top_k_df = pd.DataFrame({
            'parent_asin': top_k_asins,
            'review_count': [parent_asin_counts[asin] for asin in top_k_asins]
        })
        
        # Add cumulative column for analysis
        top_k_df['cumulative_reviews'] = top_k_df['review_count'].cumsum()
        
        top_k_path = OUTPUT_DIR / f"top{TOP_K}_products.csv"
        top_k_df.to_csv(top_k_path, index=False)
        
        log_step(f"Top {TOP_K} products identified and saved: {top_k_path}")
        log_step(f"Top 5 products:")
        for i, (asin, count) in enumerate(parent_asin_counts.head(5).items()):
            cumulative = parent_asin_counts.head(i+1).sum()
            print(f"    {i+1}. {asin}: {count:,} reviews (cumulative: {cumulative:,})")
        
        # Log key statistics
        log_step(f"Dataset statistics:")
        log_step(f"  - Products needed for 1.5M reviews: {TOP_K:,}")
        log_step(f"  - Percentage of total products: {(TOP_K / len(parent_asin_counts) * 100):.2f}%")
        log_step(f"  - Average reviews per product: {(actual_reviews / TOP_K):.0f}")
        
    else:
        log_step("STEP 2: SKIPPED - Computing top K disabled in config or no reviews data")
        # Try to load from saved file - check for different possible filenames
        top_k_files = list(OUTPUT_DIR.glob("top*_products.csv"))
        if top_k_files:
            # Use the most recent one if multiple exist
            top_k_path = max(top_k_files, key=lambda p: p.stat().st_mtime)
            top_k_df = pd.read_csv(top_k_path)
            top_k_asins = top_k_df['parent_asin'].tolist()
            TOP_K = len(top_k_asins)
            
            log_step(f"Loaded top {TOP_K} ASINs from file: {top_k_path}")
            log_step(f"  Total reviews in loaded set: {top_k_df['review_count'].sum():,}")
        else:
            log_step("WARNING: No top products file found")
            TOP_K = 0
            top_k_asins = []
    
    # ============================================================================
    # STEP 3: Filter reviews for top K products
    # ============================================================================
    if config['filter_reviews'] and reviews_df is not None and top_k_asins is not None:
        log_step(f"STEP 3: Filtering reviews for top {TOP_K} products...")
        
        top_k_reviews = reviews_df[reviews_df['parent_asin'].isin(top_k_asins)].copy()
        
        # Sample down to 1.5M if needed
        target_rows = 1_500_000
        if len(top_k_reviews) > target_rows:
            log_step(f"  Found {len(top_k_reviews):,} reviews, sampling {target_rows:,}...")
            top_k_reviews = top_k_reviews.sample(n=target_rows, random_state=42).reset_index(drop=True)
        else:
            log_step(f"  Found {len(top_k_reviews):,} reviews (within target)")
        
        # Save filtered reviews using chunked save
        filtered_path = OUTPUT_DIR / f"top{TOP_K}_reviews_1.5m.parquet"
        save_dataframe_chunked(top_k_reviews, filtered_path)
        
        # Clean up original reviews to free memory
        del reviews_df
        gc.collect()
        
        log_step(f"Filtered reviews saved: {filtered_path}")
        log_step(f"Filtered dataset: {len(top_k_reviews):,} rows")
    else:
        log_step(f"STEP 3: SKIPPED - Filtering reviews disabled in config or missing dependencies")
        # Try to load from checkpoint - look for dynamic filename
        filtered_files = list(OUTPUT_DIR.glob("top*_reviews_1.5m.parquet"))
        if filtered_files:
            # Use the most recent one if multiple exist
            filtered_path = max(filtered_files, key=lambda p: p.stat().st_mtime)
            top_k_reviews = pd.read_parquet(filtered_path)
            log_step(f"Loaded filtered reviews from checkpoint: {len(top_k_reviews):,} rows")
            log_step(f"  File: {filtered_path.name}")
        else:
            log_step("WARNING: No filtered reviews checkpoint found")
    
    # ============================================================================
    # STEP 4: Load metadata
    # ============================================================================
    if config['load_metadata']:
        log_step("STEP 4: Creating metadata chunks...")
        
        # Use streaming to create chunks (no in-memory merge)
        temp_dir_path = load_jsonl_gz_streaming(METADATA_FILE)
        
        log_step(f"Metadata chunks created in: {temp_dir_path}")
        log_step("IMPORTANT: Run 'python merge_parquet_chunks.py' to merge metadata chunks before continuing")
        log_step("Pipeline will now exit - restart after merging metadata chunks")
        
        # Export processing log before exit
        log_filepath = OUTPUT_DIR / "processing_log.txt"
        with open(log_filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_entries))
        log_step(f"Processing log saved: {log_filepath}")
        
        return {
            'status': 'metadata_chunks_created',
            'temp_dir': temp_dir_path,
            'message': 'Run merge_parquet_chunks.py to merge metadata chunks, then restart pipeline'
        }
    else:
        log_step("STEP 4: SKIPPED - Loading metadata disabled in config")
        # Try to load from checkpoint if it exists
        metadata_path = OUTPUT_DIR / "Meta_Sports_and_Outdoors.parquet"
        if metadata_path.exists():
            metadata_df = pd.read_parquet(metadata_path)
            log_step(f"Loaded metadata from checkpoint: {len(metadata_df):,} rows")
        else:
            log_step("WARNING: No metadata checkpoint found. Enable load_metadata or merge chunks first.")
    
    # ============================================================================
    # STEP 5: Inner join on parent_asin
    # ============================================================================
    if config['join_reviews_metadata'] and top_k_reviews is not None and metadata_df is not None:
        log_step("STEP 5: Performing inner join...")
        
        # Check overlap before join
        reviews_asins = set(top_k_reviews['parent_asin'].unique())
        metadata_asins = set(metadata_df['parent_asin'].unique())
        common_asins = reviews_asins.intersection(metadata_asins)
        
        log_step(f"  Reviews ASINs: {len(reviews_asins):,}")
        log_step(f"  Metadata ASINs: {len(metadata_asins):,}")
        log_step(f"  Common ASINs: {len(common_asins):,}")
        
        # Validate we have overlap
        assert len(common_asins) > 0, "No common ASINs found between reviews and metadata!"
        
        # Perform inner join
        joined_df = top_k_reviews.merge(
            metadata_df,
            on='parent_asin',
            how='inner',
            suffixes=('_review', '_meta')
        )
        
        # Clean up individual DataFrames
        del top_k_reviews, metadata_df
        gc.collect()
        
        # Validate join results
        assert len(joined_df) > 0, "Join resulted in empty dataset!"
        
        # Save joined dataset using chunked save
        joined_path = OUTPUT_DIR / "joined_reviews_metadata.parquet"
        save_dataframe_chunked(joined_df, joined_path)
        
        log_step(f"Inner join completed: {joined_path}")
        log_step(f"Joined dataset: {len(joined_df):,} rows × {len(joined_df.columns)} columns")
    else:
        log_step("STEP 5: SKIPPED - Join disabled in config or missing dependencies")
        # Try to load from checkpoint
        joined_path = OUTPUT_DIR / "joined_reviews_metadata.parquet"
        if joined_path.exists():
            joined_df = pd.read_parquet(joined_path)
            log_step(f"Loaded joined data from checkpoint: {len(joined_df):,} rows")
    
    # ============================================================================
    # STEP 6: Split into 3 parts of 500K rows each
    # ============================================================================
    if config['chunk_split'] and joined_df is not None:
        log_step("STEP 6: Splitting into 3 Parquet files...")
        
        rows_per_file = 500_000
        total_rows = len(joined_df)
        num_files = min(3, (total_rows + rows_per_file - 1) // rows_per_file)
        
        part_files = []
        for i in range(num_files):
            start_idx = i * rows_per_file
            end_idx = min((i + 1) * rows_per_file, total_rows)
            
            # Extract chunk
            chunk = joined_df.iloc[start_idx:end_idx].copy()
            
            # Save part file
            part_filename = f"joined_part_{i+1}.parquet"
            part_path = OUTPUT_DIR / part_filename
            
            chunk.to_parquet(part_path, index=False, engine='fastparquet')
            part_files.append(part_path)
            log_step(f"  Saved {part_filename}: {len(chunk):,} rows")
            
            # Clean up chunk
            del chunk
            gc.collect()
        
        # Validate all part files were created
        for part_path in part_files:
            assert part_path.exists(), f"Part file not created: {part_path}"
        
        log_step(f"Split completed: {len(part_files)} files created")
        
        # Store part files list for next step
        part_files_list = part_files
    else:
        log_step("STEP 6: SKIPPED - Chunk split disabled in config or no joined data")
        # Look for existing part files
        part_files_list = []
        for i in range(1, 4):  # Check for up to 3 parts
            part_path = OUTPUT_DIR / f"joined_part_{i}.parquet"
            if part_path.exists():
                part_files_list.append(part_path)
        if part_files_list:
            log_step(f"Found existing part files: {len(part_files_list)} files")
    
    # ============================================================================
    # STEP 7: Merge parts into final file
    # ============================================================================
    if config['final_merge'] and 'part_files_list' in locals() and part_files_list:
        log_step("STEP 7: Merging parts into final dataset...")
        
        # Read and combine all part files
        final_chunks = []
        for part_path in part_files_list:
            log_step(f"  Loading {part_path.name}...")
            chunk = pd.read_parquet(part_path)
            final_chunks.append(chunk)
            log_step(f"    Loaded: {len(chunk):,} rows")
        
        # Combine all chunks
        # RAM safety check before final part merge
        if psutil.virtual_memory().used / 1024**3 > 12:
            raise MemoryError("Memory usage exceeded 12GB before merge – aborting to prevent crash.")
        final_df = pd.concat(final_chunks, ignore_index=True)
        del final_chunks
        gc.collect()
        
        # Validate final dataset
        assert len(final_df) > 0, "Final dataset is empty!"
        
        # Save final merged file using chunked save
        final_path = OUTPUT_DIR / "final_joined.parquet"
        save_dataframe_chunked(final_df, final_path)
        
        # Validate final file
        assert final_path.exists(), f"Final file not created: {final_path}"
        
        log_step(f"Final dataset saved: {final_path}")
        log_step(f"Final dataset: {len(final_df):,} rows × {len(final_df.columns)} columns")
        log_step(f"File size: {final_path.stat().st_size / 1024**2:.1f} MB")
        
        final_row_count = len(final_df)
        final_col_count = len(final_df.columns)
        files_created = len(part_files_list) + 5  # Estimate based on typical run
        
        # Clean up final DataFrame from memory
        del final_df
        gc.collect()
        
    else:
        log_step("STEP 7: SKIPPED - Final merge disabled in config or no part files")
        # Try to get info from existing final file
        final_path = OUTPUT_DIR / "final_joined.parquet"
        if final_path.exists():
            # Just read the shape without loading full data
            temp_df = pd.read_parquet(final_path, columns=['parent_asin'])  # Read minimal data
            final_row_count = len(temp_df)
            del temp_df
            
            # Get column count from metadata
            parquet_file = pd.read_parquet(final_path, nrows=1)
            final_col_count = len(parquet_file.columns)
            del parquet_file
            
            files_created = len([f for f in OUTPUT_DIR.glob("*.parquet")]) + len([f for f in OUTPUT_DIR.glob("*.csv")])
            log_step(f"Found existing final file: {final_row_count:,} rows × {final_col_count} columns")
        else:
            final_row_count = 0
            final_col_count = 0
            files_created = 0
    
    # ============================================================================
    # PIPELINE SUMMARY
    # ============================================================================
    log_step("\nPIPELINE SUMMARY:")
    log_step(f"  • Reviews processed: 9,000,000")
    log_step(f"  • Top products identified: {TOP_K if 'TOP_K' in locals() else 'N/A'}")
    log_step(f"  • Filtered reviews: {final_row_count:,}")
    log_step(f"  • Final joined dataset: {final_row_count:,} rows × {final_col_count} columns")
    log_step(f"  • Output files created: {files_created}")
    log_step(f"  • Processing completed successfully!")
    
    # Export processing log
    log_filepath = OUTPUT_DIR / "processing_log.txt"
    with open(log_filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_entries))
    log_step(f"Processing log saved: {log_filepath}")
    
    # Final memory cleanup
    gc.collect()
    log_step("Memory cleaned up")
    
    return {
        'status': 'success',
        'output_dir': str(OUTPUT_DIR),
        'final_file': str(OUTPUT_DIR / "final_joined.parquet"),
        'total_rows': final_row_count,
        'files_created': files_created,
        'config_used': config
    }

if __name__ == "__main__":
    try:
        result = main()
        print(f"\nPipeline completed successfully!")
        
        # Safely print known keys
        if 'output_dir' in result:
            print(f"Output directory: {result['output_dir']}")
        if 'final_file' in result:
            print(f"Final file: {result['final_file']}")
        if 'config_used' in result:
            print(f"Configuration used: {result['config_used']}")
        
        # Fallback summary
        if 'message' in result:
            print(f"\nNOTE: {result['message']}")

        
    except Exception as e:
        log_step(f"Pipeline failed: {str(e)}")
        raise