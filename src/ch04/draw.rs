use plotters::prelude::*;

use crate::ch04::gradient::numerical_diff;

pub struct DrawContent<F>
where
    F: Fn(f64) -> f64 + Copy,
{
    pub function: F,
    pub caption: String,
}

pub fn draw_graph<F>(content: DrawContent<F>) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(f64) -> f64 + Copy,
{
    let x = (-1, 20);
    let root = BitMapBackend::new("plotters/0.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let x_range = x.0 as f64..x.1 as f64;
    let y_range = -0.1f64..6f64;
    let f = content.function;

    let mut chart = ChartBuilder::on(&root)
        .caption(content.caption, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        (100 * x.0..100 * x.1)
            .map(|x| x as f64 / 100.0)
            .map(|x| (x, f(x))),
        &BLUE,
    ))?;
    // .label("y = x^2")
    // .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.draw_series(LineSeries::new(
        (100 * x.0..100 * x.1)
            .map(|x| x as f64 / 100.0)
            .map(|x| (x, numerical_diff(f, 5.) * (x - 5.) + f(5.))),
        &RED,
    ))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
