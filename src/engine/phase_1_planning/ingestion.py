import shutil
import polars as pl
from pathlib import Path

class DataIngestion:
    """
    Production Ingestion Engine utilizing Polars.
    Copies raw analytical files into 'data/raw/', applies structural 
    safety caps, and saves them into 'data/active/' for downstream AI processing.
    """
    def __init__(self, workspace_dir: str):
        self.workspace_path = Path(workspace_dir)
        
        # Source folder (Raw Data)
        self.raw_dir = self.workspace_path / "data" / "raw"
        # Destination folder (AI processing zone)
        self.active_dir = self.workspace_path / "data" / "active"
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.active_dir.mkdir(parents=True, exist_ok=True)
        
    def ingest_csv(self, file_path_or_name: str, n_rows: int = None, n_cols: int = None) -> str:
        """
        Locates the user's dataset (either locally or globally), copies it to raw/,
        enforces safety dimensions, and saves a prefixed active copy.
        """
        input_path = Path(file_path_or_name)
        filename = input_path.name
        raw_path = self.raw_dir / filename
        
        # 1. Resolve the raw file location (Auto-Copy Feature)
        if input_path.is_absolute() or input_path.exists():
            # If the user passed a valid external path, copy it into our raw folder
            if not raw_path.exists() or input_path.resolve() != raw_path.resolve():
                print(f"📥 Importing external dataset into workspace: {filename}")
                shutil.copy2(input_path, raw_path)
        
        # Guardrail: Verify the file exists in the raw directory
        if not raw_path.exists():
            raise FileNotFoundError(
                f"[-] Execution Halted: Target data file '{filename}' was not found.\n"
                f"> Please verify that your dataset is located at: {raw_path.resolve()}\n"
                f"> Or provide an absolute path to the file."
            )
            
        print(f"⚙️ Loading dataset via Polars for active processing...")
        
        try:
            # 2. Read directly from disk (Polars applies n_rows cap instantly)
            df = pl.read_csv(raw_path, n_rows=n_rows)
            
            # 3. Enforce column boundary layout limits if requested
            if n_cols is not None and n_cols < len(df.columns):
                target_columns = df.columns[:n_cols]
                df = df.select(target_columns)
                print(f"⚠️ System Boundary: Retained the first {n_cols} dataset columns.")
            
            # 4. Create dynamic target filename and save to active_data folder
            output_filename = f"copy_{filename}"
            active_path = self.active_dir / output_filename
            
            df.write_csv(active_path)
            
            print(f"✅ Ingestion complete. AI working copy created: data/active/{output_filename}")
            print(f"📊 Dimensions: {df.shape[0]} rows x {df.shape[1]} columns.")
            
            return str(active_path.resolve())
            
        except Exception as e:
            print(f"❌ Critical error during Polars ingestion process: {e}")
            raise e