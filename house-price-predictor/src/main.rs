use std::str::from_utf8_mut;

use anyhow::Ok;
use polars::frame::DataFrame;
use house_price_predictor :: {download_csv_file, load_csv_file, 
    push_model_to_s3, split_features_and_targets, train_test_split,
     train_xgboost_model};
use tokio::runtime::{self, Runtime};

fn main() -> anyhow::Result<()> {
    println!("Starting training script ...");

    let csv_file_path: String = download_csv_file()?;

    let df: DataFrame = load_csv_file(&csv_file_path)?;

    let (train_df, test_df) = train_test_split(&df, 0.2)?;

    let (x_train, y_train) = split_features_and_targets(&train_df)?;

    let (x_test, y_test) = split_features_and_targets(&test_df)?;

    let path_to_model: String = train_xgboost_model(&x_train, &y_train, &x_test, &y_test)?;

    let runtime: Runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(push_model_to_s3(&path_to_model))?;


    println!("Model pushed to S3 bucket");
    Ok(())
}

