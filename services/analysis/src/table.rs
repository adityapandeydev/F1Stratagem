//! Column access over a Parquet file read with arrow-rs.
//!
//! Archive tables are written by pyarrow; columns are cast on read so callers get
//! plain Rust values whatever physical type (Utf8 or LargeUtf8, Int32 or Int64) a
//! file happens to use.

use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use arrow_array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_cast::cast;
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub struct Table {
    batches: Vec<RecordBatch>,
}

impl Table {
    pub fn read(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("reading {}", path.display()))?
            .build()?;
        let batches = reader.collect::<Result<Vec<_>, _>>()?;
        Ok(Self { batches })
    }

    pub fn len(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn has(&self, name: &str) -> bool {
        self.batches.first().is_some_and(|b| b.schema().index_of(name).is_ok())
    }

    fn arrays(&self, name: &str, to: &DataType) -> Result<Vec<ArrayRef>> {
        self.batches
            .iter()
            .map(|b| {
                let column = b.column_by_name(name).ok_or_else(|| anyhow!("no column {name}"))?;
                Ok(cast(column, to)?)
            })
            .collect()
    }

    pub fn i64s(&self, name: &str) -> Result<Vec<Option<i64>>> {
        let mut out = Vec::with_capacity(self.len());
        for a in self.arrays(name, &DataType::Int64)? {
            let a = a.as_any().downcast_ref::<Int64Array>().expect("cast to Int64");
            out.extend((0..a.len()).map(|i| a.is_valid(i).then(|| a.value(i))));
        }
        Ok(out)
    }

    pub fn f64s(&self, name: &str) -> Result<Vec<Option<f64>>> {
        let mut out = Vec::with_capacity(self.len());
        for a in self.arrays(name, &DataType::Float64)? {
            let a = a.as_any().downcast_ref::<Float64Array>().expect("cast to Float64");
            out.extend((0..a.len()).map(|i| (a.is_valid(i) && !a.value(i).is_nan()).then(|| a.value(i))));
        }
        Ok(out)
    }

    pub fn strings(&self, name: &str) -> Result<Vec<Option<String>>> {
        let mut out = Vec::with_capacity(self.len());
        for a in self.arrays(name, &DataType::Utf8)? {
            let a = a.as_any().downcast_ref::<StringArray>().expect("cast to Utf8");
            out.extend((0..a.len()).map(|i| a.is_valid(i).then(|| a.value(i).to_string())));
        }
        Ok(out)
    }

    pub fn bools(&self, name: &str) -> Result<Vec<Option<bool>>> {
        let mut out = Vec::with_capacity(self.len());
        for a in self.arrays(name, &DataType::Boolean)? {
            let a = a.as_any().downcast_ref::<BooleanArray>().expect("cast to Boolean");
            out.extend((0..a.len()).map(|i| a.is_valid(i).then(|| a.value(i))));
        }
        Ok(out)
    }
}
