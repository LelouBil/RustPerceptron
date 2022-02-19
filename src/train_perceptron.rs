use std::ops::{Add, Deref, Mul};
use itertools::Itertools;
use ndarray::{Array2, ArrayBase, ArrayView1, ArrayView2, Axis, concatenate, Dim, Ix, Ix2, OwnedRepr, Shape, ShapeBuilder, ShapeError, StrideShape};
use crate::record::{PerceptronDataset, PerceptronKnowlege};


pub fn train_perceptron(dataset: &PerceptronDataset) -> anyhow::Result<PerceptronKnowlege> {
    let colonne_uns: Array2<f64> = Array2::from_elem((dataset.records.len(), 1), 1.0);
    let (X, Y) = to_matrix(dataset).map(|(x, y)| (x.mapv(|e| *e), y.mapv(|e| e as f64)))?;


    let concated: Array2<f64> = concatenate(Axis(1), &[colonne_uns.view(), X.view()])?
        .into_shape((5,3))?;
    
    let mut w: Array2<f64> = Array2::ones((concated.ncols(), 1));
    


    loop {
        let prediction: Array2<f64> = concated.dot(&w) * &Y;
        let pabien : Option<((usize,usize),&f64)> = prediction.indexed_iter()
            .find(|((i,j),elem) : &((usize,usize),&f64)| elem < &&0.0);

        if let Some(((i, j), value)) = pabien {
            //entraine
            let actual = &Y[[i,0]];
            let row : ArrayView2<f64> = concated.row(i).insert_axis(Axis(1));
            w.scaled_add(*actual, &row);
        }
        else {
            break;
        }
    }

    let kn = PerceptronKnowlege { vector: w.remove_axis(Axis(1)) };
    Ok(kn)
}

pub fn to_matrix(dataset: &PerceptronDataset) -> Result<(Array2<&f64>, Array2<i8>), ShapeError> {
    let x = Array2::from_shape_vec(
        (dataset.records.len(), dataset.features_lenght),
        dataset.records.iter()
            .flat_map(|record| &record.features)
            .collect_vec(),
    )?;

    let y = Array2::from_shape_vec((dataset.records.len(), 1).into_shape(),
                                   dataset.records.iter()
                                       .map(|rec| rec.label.val())
                                       .collect_vec(),
    )?;

    Ok((x, y))
}