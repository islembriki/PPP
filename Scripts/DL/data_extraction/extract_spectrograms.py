import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import sys

# Add paths module
sys.path.insert(0, os.path.dirname(__file__))
from paths import DOWNLOAD_FOLDER, EXTRACTED_DATA_BASE, ZIP_FILES, create_directories

# ════════════════════════════════════════════════════════════════
# ZIP EXTRACTION SCRIPT FOR DRONE SPECTROGRAMS (WITH PROGRESS)
# ════════════════════════════════════════════════════════════════

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0

def extract_zip_with_progress(zip_path, extract_to):
    """
    Extracts a ZIP file with progress bar showing:
    - Percentage complete
    - Files extracted count
    - Time remaining (if available)

    Args:
        zip_path: Full path to the ZIP file
        extract_to: Directory where files will be extracted

    Returns:
        tuple: (success: bool, file_count: int, error_msg: str or None)
    """
    try:
        print(f"\n📦 Extracting: {os.path.basename(zip_path)}")
        file_size = get_file_size_mb(zip_path)
        print(f"   📊 Size: {file_size:.2f} MB")
        print(f"   📁 To: {extract_to}\n")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files to extract
            file_list = zip_ref.namelist()
            total_files = len(file_list)

            print(f"   📈 Total files in ZIP: {total_files}")

            # Create progress bar
            with tqdm(total=total_files, desc="Extracting", unit="file",
                      ncols=80, colour="green") as pbar:
                for file in file_list:
                    zip_ref.extract(file, extract_to)
                    pbar.update(1)

        # Count extracted PNG files (ignore directory structure)
        png_count = 0
        for root, dirs, files in os.walk(extract_to):
            png_count += len([f for f in files if f.lower().endswith('.png')])

        print(f"\n✅ Extraction successful!")
        print(f"   📊 PNG files extracted: {png_count}")
        print(f"   📁 Location: {extract_to}\n")

        return True, png_count, None

    except FileNotFoundError as e:
        error_msg = f"ZIP file not found: {zip_path}"
        print(f"❌ {error_msg}")
        return False, 0, error_msg

    except zipfile.BadZipFile as e:
        error_msg = f"Invalid ZIP file (corrupted): {zip_path}"
        print(f"❌ {error_msg}")
        return False, 0, error_msg

    except PermissionError as e:
        error_msg = f"Permission denied accessing: {zip_path}"
        print(f"❌ {error_msg}")
        return False, 0, error_msg

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        return False, 0, error_msg

def main():
    """Main extraction workflow"""

    print("=" * 80)
    print("🚀 DRONE SPECTROGRAM ZIP EXTRACTION SCRIPT")
    print("=" * 80)

    # Create necessary directories
    print("\n📁 Creating directories...")
    create_directories()

    print(f"📁 Download folder: {DOWNLOAD_FOLDER}")
    print(f"📁 Extract base path: {EXTRACTED_DATA_BASE}\n")

    # Track results
    results = {}
    success_count = 0
    failed_count = 0
    not_found_count = 0
    total_files_extracted = 0

    # Process each ZIP file
    for drone_type, zip_filename in ZIP_FILES.items():
        zip_path = os.path.join(DOWNLOAD_FOLDER, zip_filename)
        extract_to = os.path.join(EXTRACTED_DATA_BASE, drone_type)

        print(f"\n{'─' * 80}")
        print(f"🔄 Processing: {drone_type}")
        print(f"{'─' * 80}")

        # Check if ZIP exists
        if not os.path.exists(zip_path):
            print(f"⏳ Not yet downloaded: {zip_filename}")
            print(f"   Expected at: {zip_path}")
            results[drone_type] = {
                "status": "WAITING",
                "files": 0,
                "error": "ZIP file not found"
            }
            not_found_count += 1
            continue

        # Create extraction directory
        os.makedirs(extract_to, exist_ok=True)

        # Extract ZIP with progress
        success, file_count, error_msg = extract_zip_with_progress(zip_path, extract_to)

        if success:
            results[drone_type] = {
                "status": "SUCCESS",
                "files": file_count,
                "path": extract_to
            }
            success_count += 1
            total_files_extracted += file_count
        else:
            results[drone_type] = {
                "status": "FAILED",
                "files": 0,
                "error": error_msg
            }
            failed_count += 1

    # Print summary
    print(f"\n{'=' * 80}")
    print("📊 EXTRACTION SUMMARY")
    print(f"{'=' * 80}\n")

    for drone_type, result in results.items():
        status = result["status"]
        files = result["files"]

        if status == "SUCCESS":
            print(f"✅ {drone_type:15s}: {files:8d} PNG files extracted")
        elif status == "WAITING":
            print(f"⏳ {drone_type:15s}: Waiting for download")
        else:
            print(f"❌ {drone_type:15s}: {result['error']}")

    print(f"\n{'─' * 80}")
    print(f"✅ Successfully extracted: {success_count}")
    print(f"❌ Failed extractions:     {failed_count}")
    print(f"⏳ Not yet downloaded:     {not_found_count}")
    print(f"📊 TOTAL PNG FILES:        {total_files_extracted}")
    print(f"\n📁 All data extracted to: {EXTRACTED_DATA_BASE}")
    print(f"{'=' * 80}\n")

    return success_count, failed_count, not_found_count

if __name__ == "__main__":
    main()