use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::read;
use csv::ErrorKind;
use itertools::{Chunk, Itertools};
use itertools::IntoChunks;
use crate::record;
use crate::record::{PerceptronDataset, PerceptronLabel, PerceptronRecord};
use std::iter::{IntoIterator, Map};
use std::num::ParseFloatError;
use std::str::FromStr;
use anyhow::{anyhow, Context};
use thiserror::Error;


pub fn load_data_from_csv(file_name: &str) -> anyhow::Result<PerceptronDataset> {
    let mut reader = csv::Reader::from_path(file_name).map_err(Box::new)
        .with_context(|| format!("nom du fichier : {file_name}"))?;

    let var: Vec<PerceptronRecord> = reader
        .records()
        .enumerate()
        .map(|(line_num,line)| {
            let line: Vec<f64> = line
                .with_context(|| format!("a la lecture de la ligne {line_num}"))?
                
                .iter()
                .enumerate()
                
                .map(|(col_num,rec)| { rec.parse::<f64>() // on parse chaque ligne en float
                    .with_context(||format!("a lecture de la ligne {line_num} et de la colonne {col_num} (actuel {rec})"))
                }) 
                
                .collect::<Vec<anyhow::Result<_,_>>>()
                .into_iter().collect::<anyhow::Result<Vec<_>,_>>()?; // Vec<Result> en Result<Vec>
            
            match line[..] {
                [ref start @ .., end] => Ok((start.to_owned(),end)),
                _ => Err(anyhow!("Pas assez d'éléments sur la ligne (il en faut au moins 2) (actuel {})",line.len()))
            }
        })
        .map(|res : anyhow::Result<(Vec<f64>,f64)>| {
            let (data,label) = res?;
            Ok(PerceptronRecord{
                features: data,
                label: label.into()
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let features_lenght = var.first().ok_or(anyhow!("Aucun élément !"))?.features.len();
    Ok(PerceptronDataset{
        features_lenght,
        records: var
    })
}