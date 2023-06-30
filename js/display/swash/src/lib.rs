use swash::{
    scale::ScaleContext,
    shape::{Direction, ShapeContext},
    text::Script,
    zeno::Verb,
    FontDataRef,
};
use wasm_bindgen::prelude::*;

// Install rust
// Ensure we have wasm32 target with `rustup target add wasm32-unknown-unknown`
// Install wasm-pack
// Build with: `wasm-pack build --target web swash-tests`

#[wasm_bindgen]
pub struct SwashFont {
    data: Vec<u8>,
}

#[wasm_bindgen]
impl SwashFont {
    #[wasm_bindgen(constructor)]
    pub fn new(data: js_sys::Uint8Array) -> SwashFont {
        SwashFont {
            data: data.to_vec(),
        }
    }

    pub fn get_units_per_em(&self) -> u16 {
        let font = FontDataRef::new(&self.data).unwrap().get(0).unwrap();

        let mut context = ShapeContext::new();

        context.builder(font).build().metrics().units_per_em
    }

    pub fn shape_text(&self, text: &str, is_ltr: bool) -> String {
        let font = FontDataRef::new(&self.data).unwrap().get(0).unwrap();

        let mut context = ShapeContext::new();

        // TODO: cache shapers, it's just difficult with the lifetime stuff
        let mut shaper_builder = context.builder(font);
        shaper_builder = shaper_builder.script(Script::Latin);
        shaper_builder = shaper_builder.direction(if is_ltr {
            Direction::LeftToRight
        } else {
            Direction::RightToLeft
        });

        // TODO: do we need to kern?
        // shaper_builder = shaper_builder.features(&[("kern", 1)]);

        // font.features().into_iter().for_each(|feature| {
        //     if let Some(name) = feature.name() {
        //         web_sys::console::log_1( &JsValue::from( format!( "feature name: {}", name ) ) );
        //     }
        //     // tag (u32), name (option static string), action
        //     // shaper_builder = shaper_builder.features(feature);
        // });

        let mut shaper = shaper_builder.build();

        shaper.add_str(text);

        let mut result = String::new();
        result.push_str("[");
        let mut is_first = true;
        shaper.shape_with(|glyph_cluster| {
            for glyph in glyph_cluster.glyphs {
                if !is_first {
                    result.push_str(",");
                }
                is_first = false;
                result.push_str(&format!(
                    "{{\"id\":{},\"x\":{},\"y\":{},\"adv\":{}}}",
                    glyph.id, glyph.x, glyph.y, glyph.advance
                ));
            }
        });
        result.push_str("]");
        result
    }
    pub fn get_glyph(&self, id: u16, embolden_x: f32, embolden_y: f32) -> String {
        let font = FontDataRef::new(&self.data).unwrap().get(0).unwrap();

        let mut context = ScaleContext::new();
        let mut scaler = context.builder(font).hint(false).build();

        let mut result = String::new();
        if let Some(mut outline) = scaler.scale_outline(id) {
            if embolden_x != 0.0 || embolden_y != 0.0 {
                outline.embolden(embolden_x, embolden_y);
            }
            let verbs = outline.verbs();
            let points = outline.points();

            let mut point_index = 0;
            for verb in verbs {
                match verb {
                    Verb::MoveTo => {
                        let p = points[point_index];
                        point_index += 1;
                        result.push_str(&format!("M {} {} ", p.x, p.y));
                    }
                    Verb::LineTo => {
                        let p = points[point_index];
                        point_index += 1;
                        result.push_str(&format!("L {} {} ", p.x, p.y));
                    }
                    Verb::QuadTo => {
                        let p1 = points[point_index];
                        let p2 = points[point_index + 1];
                        point_index += 2;
                        result.push_str(&format!("Q {} {} {} {} ", p1.x, p1.y, p2.x, p2.y));
                    }
                    Verb::CurveTo => {
                        let p1 = points[point_index];
                        let p2 = points[point_index + 1];
                        let p3 = points[point_index + 2];
                        point_index += 3;
                        result.push_str(&format!(
                            "C {} {} {} {} {} {}",
                            p1.x, p1.y, p2.x, p2.y, p3.x, p3.y
                        ));
                    }
                    Verb::Close => {
                        result.push_str(&format!("Z "));
                    }
                }
            }
        } else {
            result.push_str("MISSING");
        }
        result
    }
}

#[wasm_bindgen(start)]
fn run() {}
