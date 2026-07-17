# Generates a clean, token-efficient summary of the provided data
# Later to be fed to the LLM as a context along with user prompt (data science problem)

import polars as pl
import json

CHARS_PER_TOKEN = 4

class DataProfiler:
    """
    Token-Efficient Data Overview Generator.
    Reads a CSV via LazyFrame, computes statistical profiles, and budgets token output.
    """
    SECTION_PRIORITY = [
        "dataset",           
        "columns",           
        "target_analysis",  
        "quality_warnings",
        "role_hints",
        "sample_rows",       
    ]

    def __init__(self, csv_path: str, target_column: str = None, sample_rows: int = 3):
        self.path = csv_path
        self.target_column = target_column
        self.sample_n = sample_rows
        self.top_k = 5
        
        # Lazy load for memory safety
        self.lf = pl.scan_csv(csv_path, infer_schema_length=5000, try_parse_dates=True)
        self.schema = self.lf.collect_schema()
        self.n_rows: int = int(self.lf.select(pl.len()).collect().item())

    def generate(self, token_budget: int = 1500) -> str:
        """Build the profile and trim to fit token budget."""
        profile = self.profile_dict()
        return self._fit(profile, token_budget)

    def profile_dict(self) -> dict:
        """Full structured profile."""
        cols = self._column_profiles()
        profile_data = {
            "dataset": self._dataset_overview(),
            "columns": cols,
            "quality_warnings": self._quality_warnings(cols),
            "role_hints": self._role_hints(cols),
            "sample_rows": self._sample_rows(),
        }
        
        # Inject target analysis if a target column was provided
        if self.target_column and self.target_column in self.schema:
            profile_data["target_analysis"] = self._get_target_balance()
            
        return profile_data

    def _dataset_overview(self) -> dict:
        return {
            "path": self.path,
            "rows": self.n_rows,
            "columns": len(self.schema)
        }

    def _column_profiles(self) -> dict:
        out: dict[str, dict] = {}
        base = self.lf.select(
            [pl.col(c).null_count().alias(f"{c}__nulls") for c in self.schema]
            + [pl.col(c).n_unique().alias(f"{c}__uniq") for c in self.schema]
        ).collect()

        for name, dtype in self.schema.items():
            nulls = int(base[f"{name}__nulls"][0])
            uniq = int(base[f"{name}__uniq"][0])
            info: dict = {
                "dtype": str(dtype),
                "null_pct": round(100 * nulls / max(self.n_rows, 1), 2),
                "n_unique": uniq,
            }
            try:
                if dtype.is_numeric():
                    info.update(self._numeric_stats(name))
                elif dtype.is_temporal():
                    info.update(self._temporal_stats(name))
                else:
                    info.update(self._string_stats(name, uniq))
            except Exception as e:
                info["profile_error"] = str(e)[:80]
            out[name] = info
        return out

    def _numeric_stats(self, c: str) -> dict:
        col = pl.col(c)
        q = self.lf.select(
            col.min().alias("min"), col.max().alias("max"),
            col.mean().alias("mean"), col.std().alias("std"),
            col.median().alias("p50"), col.skew().alias("skew"),
        ).collect().row(0, named=True)

        def r(v): return None if v is None else round(float(v), 4)

        return {
            "min": r(q["min"]), "median": r(q["p50"]), "max": r(q["max"]),
            "mean": r(q["mean"]), "std": r(q["std"]), "skew": r(q["skew"])
        }

    def _temporal_stats(self, c: str) -> dict:
        q = self.lf.select(pl.col(c).min().alias("mn"), pl.col(c).max().alias("mx")).collect().row(0, named=True)
        return {"min": str(q["mn"]), "max": str(q["mx"])}

    def _string_stats(self, c: str, uniq: int) -> dict:
        info: dict = {}
        if 0 < uniq <= max(50, self.n_rows // 10):
            top = self.lf.group_by(c).len().sort("len", descending=True).head(self.top_k).collect()
            info["top_values"] = {str(r0)[:40]: int(r1) for r0, r1 in top.iter_rows()}
        return info

    def _get_target_balance(self) -> dict:
        """Evaluates class balance specifically for classification targets."""
        distribution = (
            self.lf.group_by(self.target_column)
            .len()
            .with_columns((pl.col("len") / self.n_rows * 100).round(2).alias("percentage"))
            .sort("len", descending=True)
            .collect()
            .to_dicts()
        )
        return {
            "target_column": self.target_column,
            "class_distribution": distribution
        }

    def _quality_warnings(self, cols: dict) -> list[str]:
        w = []
        for name, c in cols.items():
            if c["n_unique"] <= 1: w.append(f"'{name}' is constant — drop it")
            if c["null_pct"] > 40: w.append(f"'{name}' is {c['null_pct']}% null — impute/drop")
            skew = c.get("skew")
            if skew is not None and abs(skew) > 3: w.append(f"'{name}' skewed (skew={skew})")
        return w

    def _role_hints(self, cols: dict) -> dict:
        hints = {"likely_ids": [], "binary": []}
        for name, c in cols.items():
            if c["n_unique"] == self.n_rows: hints["likely_ids"].append(name)
            if c["n_unique"] == 2: hints["binary"].append(name)
        return {k: v for k, v in hints.items() if v}

    def _sample_rows(self) -> list[dict]:
        head = self.lf.head(self.sample_n).collect()
        return [{k: (str(v)[:30] if v is not None else None) for k, v in row.items()} for row in head.iter_rows(named=True)]

    def _fit(self, profile: dict, budget: int) -> str:
        included = list(self.SECTION_PRIORITY)
        while True:
            payload = {k: profile[k] for k in included if profile.get(k)}
            text = json.dumps(payload, default=str, separators=(",", ":"))
            if len(text) <= budget * CHARS_PER_TOKEN or len(included) <= 2:
                return text
            included.pop()

if __name__ == "__main__":
    import os
    # Local Testing
    test_file = "workspace/active_data/copy_gold_sales_analytics.csv"
    if os.path.exists(test_file):
        print("[+] Testing Merged Data Profiler")
        profiler = DataProfiler(test_file, target_column="CUSTOMER_SEGMENT")
        json_output = profiler.generate(token_budget=1500)
        print("\Generated Token-Efficient Profile:\n")
        print(json.dumps(json.loads(json_output), indent=2))
    else:
        print(f"[-] Test file {test_file} not found. Run ingestion.py first.")