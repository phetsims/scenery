
// Requires a var<private> pathdata_base to be set to the start of the path data.

fn read_f32_point(ix: u32) -> vec2<f32> {
    return bitcast<vec2<f32>>(vec2(scene[pathdata_base + ix], scene[pathdata_base + ix + 1u]));
}

fn read_i16_point(ix: u32) -> vec2<f32> {
    let raw = scene[pathdata_base + ix];
    let x = f32(i32(raw << 16u) >> 16u);
    let y = f32(i32(raw) >> 16u);
    return vec2(x, y);
}
