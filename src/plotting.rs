use itertools::Itertools;
use plotters::chart::{ChartBuilder, LabelAreaPosition};
use plotters::drawing::IntoDrawingArea;
use plotters::element::TriangleMarker;
use plotters::prelude::{BitMapBackend, RED};
use plotters::series::LineSeries;
use plotters::style::{BLUE, GREEN, WHITE};
use crate::record::{PerceptronDataset, PerceptronKnowlege, PerceptronLabel};

pub fn plot_results(dataset: &PerceptronDataset, trained: &PerceptronKnowlege) {
    let root_drawing_area = BitMapBackend::new("graph.png", (1024, 768)).into_drawing_area();

    root_drawing_area.fill(&WHITE).unwrap();

    let (min_x,max_x) = dataset.records.iter()
        .map(|e| e.features[0] as i32)
        .minmax().into_option().map(|(a,b)|(a - 1, b + 1)).unwrap();
    

    let (min_y,max_y) = dataset.records.iter()
        .map(|e| e.features[1] as i32).minmax().into_option().map(|(a,b)|(a - 1, b + 1)).unwrap();

    let mut chart = ChartBuilder::on(&root_drawing_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        // enable X axis, the size is 40 px
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)
        .unwrap();
    
    chart.configure_mesh().draw().unwrap();



    chart.draw_series(
        dataset.records.iter().map(|r| {
            TriangleMarker::new((r.features[0] as i32, r.features[1] as i32),
                                5,
                                match r.label {
                                    PerceptronLabel::Positive => &BLUE,
                                    PerceptronLabel::Negative => &RED
                                },
            )
        }
        )
    ).unwrap();
    
    
    // fonction
    let fw = |x: i32| {
        (-(trained.vector[0]/trained.vector[2]) - (trained.vector[1]/trained.vector[2]) * (x as f64)) as i32
    };


    
    let vect: Vec<i32> = vec![min_x as i32,max_x as i32];
    chart.draw_series(LineSeries::new(vect.iter().map(|e| (*e,fw(*e))), &GREEN)).unwrap();
}