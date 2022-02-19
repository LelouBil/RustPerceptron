mod record;
mod data_loading;
mod train_perceptron;
mod plotting;

#[macro_use]
extern crate ndarray;

use ndarray::Array2;
use crate::plotting::plot_results;


fn main()  {
    if let Err(e) = perceptron_main() {
        eprintln!("Erreure : {e:#}");
    }
}

fn perceptron_main() -> anyhow::Result<()>{
    let data = data_loading::load_data_from_csv("data.csv")?;
    let trained = train_perceptron::train_perceptron(&data)?;
    
    dbg!(&trained);
    plot_results(&data,&trained);
    
    Ok(())
}