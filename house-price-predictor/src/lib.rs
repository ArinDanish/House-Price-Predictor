use polars::prelude::*;
use anyhow::Ok;
use reqwest::blocking::Response;
use bytes::Bytes; 
use rand::thread_rng;
use rand::seq::SliceRandom;
use xgboost::{DMatrix, parameters, Booster};
use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::{Client, Error};

pub fn download_csv_file() -> anyhow::Result<String> {

    let url: &str = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv";

    let response: Response = reqwest:: blocking :: get(url)?;
    
    let bytes: Bytes = response.bytes()?;

    let filepath: &str = "boston_housing.csv";

    std::fs::write(filepath, bytes)?;

    Ok(filepath.to_string())

}


pub fn load_csv_file(filepath: &str) -> anyhow::Result<DataFrame> {
    
    let df: DataFrame = CsvReader::from_path(filepath)?.finish()?;

    println!("Loaded {} rows and {} columns", df.height(), df.width());
    println!("{}", df.head(Some(5)));

    Ok(df)
}


pub fn train_test_split(df: &DataFrame, test_size_perc: f64) -> anyhow::Result<(DataFrame, DataFrame)> {

    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..df.height()).collect();
    indices.shuffle(&mut rng);

    let split_idx: usize = (df.height() as f64 * (1.0-test_size_perc)).ceil() as usize;
    let test_indices: Vec<usize> = indices[0..split_idx].to_vec();
    let train_indices: Vec<usize> = indices[split_idx..].to_vec();

    let train_indices_ca:ChunkedArray<UInt32Type> = UInt32Chunked::from_vec(
        "",  train_indices.iter().map(|&x| x as u32).collect());

    let test_indices_ca:ChunkedArray<UInt32Type> = UInt32Chunked::from_vec(
        "", test_indices.iter().map(|&x| x as u32).collect());

    let test_df: DataFrame = df.take(&test_indices_ca)?;
    let train_df: DataFrame = df.take(&train_indices_ca)?;

    println!("Training set size: {}",train_df.height());
    println!("Test set size: {}", test_df.height());

    Ok((train_df, test_df))
}

pub fn split_features_and_targets(df: &DataFrame) -> anyhow::Result<(DataFrame, DataFrame)> {

    let feature_names: Vec<&str> = vec!["crim", "zn", "indus", "chas", "nox", "rm",
     "age", "dis", "rad", "tax", "ptratio", "b", "lstat"];
    let target_name: Vec<&str>= vec!["medv"];

    let features: DataFrame = df.select(feature_names)?;
    let target: DataFrame = df.select(target_name)?;

    Ok((features, target))
}

pub fn train_xgboost_model(
    x_train: &DataFrame,
    y_train: &DataFrame,
    x_test: &DataFrame,
    y_test: &DataFrame,
 ) -> anyhow::Result<String> {

    let x_train_array = x_train.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let y_train_array = y_train.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let x_test_array = x_test.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let y_test_array = y_test.to_ndarray::<Float32Type>(IndexOrder::C)?;

    let x_train_slice: &[f32]= x_train_array.as_slice().expect("failed to conver x_train_array to slice - array may not be contigous");
    let y_train_slice: &[f32] = y_train_array.as_slice().expect("failed to conver y_train_array to slice - array may not be contigous");
    let x_test_slice: &[f32] = x_test_array.as_slice().expect("failed to conver x_test_array to slice - array may not be contigous");
    let y_test_slice: &[f32] = y_test_array.as_slice().expect("failed to conver y_test_array to slice - array may not be contigous");

    let mut dmatrix_train = DMatrix::from_dense(x_train_slice, x_train.height())?;
    dmatrix_train.set_labels(y_train_slice)?;

    let mut dmatrix_test = DMatrix::from_dense(x_test_slice, x_test.height())?;
    dmatrix_test.set_labels(y_test_slice)?;

    let evaluation_sets = &[
        (&dmatrix_train,"train"),
        (&dmatrix_test, "test")
        ];

    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dmatrix_train)
        .evaluation_sets(Some(evaluation_sets))
        .build().unwrap();

    let model: Booster = Booster::train(&training_params).unwrap();

    println!("Test {:?}", model.predict(&dmatrix_test).unwrap());

    // Save the model to a file

    let model_path: &str = "boston_house_model.bin";
    model.save(model_path)?;

    println!("Model saved to {}", model_path);

    Ok(model_path.to_string())

 }

 pub async fn push_model_to_s3(path_to_model: &str) -> anyhow::Result<()> {

    let region_provider = RegionProviderChain::default_provider().or_else("eu-north-1");
    // Get the region from the provider chain
    //let region = region_provider.region().await;

    // Print the resolved region
    //println!("Using AWS Region: {}", region);
    //let region = region_provider.region().await;
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
    .region(region_provider)
    .load()
    .await;
      
    let client = Client::new(&config);

    let model_file_bytes = std::fs::read(path_to_model)?;

    let bucket_name: &str = "house-price-predictor-models";
    let key: &str = "boston_house_model.bin";

    let _result = client
        .put_object()
        .bucket(bucket_name)
        .key(key)
        .body(model_file_bytes.into())
        .send()
        .await?;

    println!("Model pushed to S3");

    Ok(())
 }

