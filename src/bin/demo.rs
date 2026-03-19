use rudygrad::engine::Value;
use rudygrad::nn::MLP;
use rudygrad::datasets::make_moons;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating moons dataset...");
    let (x_data, y_data) = make_moons(100, 0.1);

    let inputs: Vec<Vec<Value>> = x_data.iter()
        .map(|row| row.iter().map(|&x| Value::new(x)).collect())
        .collect();
    let targets: Vec<Value> = y_data.iter().map(|&y| Value::new(y)).collect();

    let model = MLP::new(2, vec![16, 16, 1]);

    let root = BitMapBackend::gif("demo.gif", (600, 500), 100)?.into_drawing_area();

    println!("Starting training loop...\n");
    let start_time = std::time::Instant::now();
    let mut total_training_time = std::time::Duration::from_secs(0);

    for k in 0..=100 {
        let step_start = std::time::Instant::now();
        let mut total_loss = Value::new(0.0);
        let mut correct = 0;

        for (x, y) in inputs.iter().zip(targets.iter()) {
            let score = &model.call(x.clone())[0]; 
            let margin = Value::new(1.0) - (y.clone() * score.clone());
            let loss = margin.relu();
            total_loss = total_loss + loss;

            let score_f32 = score.0.borrow().data;
            let y_f32 = y.0.borrow().data;
            if (y_f32 > 0.0 && score_f32 > 0.0) || (y_f32 < 0.0 && score_f32 < 0.0) {
                correct += 1;
            }
        }

        let final_loss = total_loss * Value::new(1.0 / inputs.len() as f32);
        let accuracy = (correct as f32 / inputs.len() as f32) * 100.0;

        model.zero_grad();
        final_loss.backward();

        let learning_rate = 1.0 - (0.9 * k as f32 / 100.0); 
        for p in model.parameters() {
            let grad = p.0.borrow().grad;
            p.0.borrow_mut().data -= learning_rate * grad;
        }
        total_training_time += step_start.elapsed();

        if k % 5 == 0 {
            println!("Step {:>3} | Loss: {:.4} | Accuracy: {:.1}%", k, final_loss.0.borrow().data, accuracy);

            root.fill(&WHITE)?;
            let mut chart = ChartBuilder::on(&root)
                .margin(10)
                .build_cartesian_2d(-1.5f32..2.5f32, -1.0f32..1.5f32)?;

            let res = 60;
            let step_x = 4.0 / res as f32;
            let step_y = 2.5 / res as f32;
            
            for i in 0..res {
                for j in 0..res {
                    let gx = -1.5 + (i as f32) * step_x;
                    let gy = -1.0 + (j as f32) * step_y;
                    
                    let pred = &model.call(vec![Value::new(gx), Value::new(gy)])[0];
                    let score = pred.0.borrow().data;
                    
                    let squash = score.tanh(); 
                    let v = (squash + 1.0) / 2.0;
                    
                    let (r, g, b) = if v < 0.5 {
                        let mix = v * 2.0;
                        (255, (140.0 + 115.0 * mix) as u8, (140.0 + 115.0 * mix) as u8)
                    } else {
                        let mix = (v - 0.5) * 2.0;
                        ((255.0 - 115.0 * mix) as u8, (255.0 - 55.0 * mix) as u8, 255)
                    };

                    chart.draw_series(std::iter::once(Rectangle::new(
                        [(gx, gy), (gx + step_x, gy + step_y)],
                        RGBColor(r, g, b).filled(),
                    )))?;
                }
            }

            for (idx, pt) in x_data.iter().enumerate() {
                let color = if y_data[idx] > 0.0 { &BLUE } else { &RED };
                chart.draw_series(PointSeries::of_element(
                    vec![(pt[0], pt[1])],
                    4,
                    color,
                    &|c, s, st| {
                        EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
                    },
                ))?;
            }
            root.present()?;
        }
    }
    
    let total_elapsed = start_time.elapsed();
    let avg_step_time = total_training_time.as_secs_f32() / 101.0; 
    
    println!("\nTraining complete!");
    println!("Total training duration:  {:.2?}", total_elapsed);
    println!("Core training time only: {:.2?}", total_training_time);
    println!("Average core step time:  {:.4}ms", avg_step_time * 1000.0);
    println!("Saved GIF to 'demo.gif'");
    Ok(())
}
