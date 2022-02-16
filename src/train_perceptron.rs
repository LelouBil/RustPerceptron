use itertools::Itertools;
use ndarray::{Array, Array2, ArrayBase, Ix2, OwnedRepr, ShapeError};
use crate::record::{PerceptronDataset, PerceptronKnowlege, PerceptronRecord};

// pub fn train_perceptron(dataset: Vec<PerceptronRecord>) -> PerceptronKnowlege{
//     let matrixed = to_matrix(dataset);
//     
// }

type RefArray2t<'a, T> = Array2t<&'a T>;
type Array2t<T> = ArrayBase<OwnedRepr<T>, Ix2>;

pub fn to_matrix(dataset: &PerceptronDataset) -> Result<(RefArray2t<f64>, Array2t<i8>), ShapeError> {
    let x = Array2::from_shape_vec(
        (dataset.records.len(), dataset.features_lenght),
        dataset.records.iter()
            .flat_map(|record| &record.features)
            .collect_vec()
    )?;

    let y = Array2::from_shape_vec((dataset.records.len(), 1),
                                   dataset.records.iter()
                                       .map(|rec| rec.label.val())
                                       .collect_vec(),
    )?;

    Ok((x, y))
}