// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Helpers for working with transforms.

struct Transform {
    matrx: vec4<f32>,
    translate: vec2<f32>,
}

fn transform_apply(transform: Transform, p: vec2<f32>) -> vec2<f32> {
    return transform.matrx.xy * p.x + transform.matrx.zw * p.y + transform.translate;
}

fn transform_inverse(transform: Transform) -> Transform {
    let inv_det = 1.0 / (transform.matrx.x * transform.matrx.w - transform.matrx.y * transform.matrx.z);
    let inv_mat = inv_det * vec4(transform.matrx.w, -transform.matrx.y, -transform.matrx.z, transform.matrx.x);
    let inv_tr = mat2x2(inv_mat.xy, inv_mat.zw) * -transform.translate;
    return Transform(inv_mat, inv_tr);
}

fn transform_mul(a: Transform, b: Transform) -> Transform {
    return Transform(
        a.matrx.xyxy * b.matrx.xxzz + a.matrx.zwzw * b.matrx.yyww,
        a.matrx.xy * b.translate.x + a.matrx.zw * b.translate.y + a.translate
    );
}

fn read_transform(transform_base: u32, ix: u32) -> Transform {
    let base = transform_base + ix * 6u;
    let matrx = bitcast<vec4<f32>>(vec4(
        scene[base],
        scene[base + 1u],
        scene[base + 2u],
        scene[base + 3u],
    ));
    let translate = bitcast<vec2<f32>>(vec2(
        scene[base + 4u],
        scene[base + 5u],
    ));
    return Transform(matrx, translate);
}
