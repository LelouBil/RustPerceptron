use crate::record::PerceptronLabel::{Negative, Positive};

#[derive(Debug)]
pub enum PerceptronLabel{
    Positive,
    Negative
}

impl PerceptronLabel {
    pub(crate) fn val(&self) -> i8 {
        match self {
            Positive => 1,
            Negative => -1
        }
    }
}

impl From<f64> for PerceptronLabel {
    fn from(num: f64) -> Self {
        if num > 0.0 { 
            Positive
        } else{
            Negative
        }
    }
}

pub struct PerceptronKnowlege{
    vector: Vec<f64>
}

#[derive(Debug)]
pub struct PerceptronDataset{
    pub features_lenght: usize,
    pub records: Vec<PerceptronRecord>
}

#[derive(Debug)]
pub struct PerceptronRecord {
    pub(crate) features: Vec<f64>,
    pub(crate) label: PerceptronLabel
}