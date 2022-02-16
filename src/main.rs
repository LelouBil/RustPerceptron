mod record;
mod data_loading;
mod train_perceptron;

fn main()  {
    if let Err(e) = perceptron_main() {
        eprintln!("Erreure : {e:#}");
    }
}

fn perceptron_main() -> anyhow::Result<()>{
    let data = data_loading::load_data_from_csv("data.csv")?;
    dbg!(&data);
    dbg!(train_perceptron::to_matrix(&data));
    Ok(())
}