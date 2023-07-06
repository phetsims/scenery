// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// Fine rasterizer. This can run in simple (just path rendering) and full
// modes, controllable by #define.

// This is a cut'n'paste w/ backdrop.
struct Tile {
    backdrop: i32,
    segments: u32,
}

#import segment
#import config

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> tiles: array<Tile>;

@group(0) @binding(2)
var<storage> segments: array<Segment>;

#ifdef full

#import blend
#import ptcl

let GRADIENT_WIDTH = 512;

@group(0) @binding(3)
var output: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(4)
var<storage> ptcl: array<u32>;

@group(0) @binding(5)
var gradients: texture_2d<f32>;

@group(0) @binding(6)
var<storage> info: array<u32>;

@group(0) @binding(7)
var image_atlas: texture_2d<f32>;

fn read_fill(cmd_ix: u32) -> CmdFill {
    let tile = ptcl[cmd_ix + 1u];
    let backdrop = i32(ptcl[cmd_ix + 2u]);
    return CmdFill(tile, backdrop);
}

fn read_stroke(cmd_ix: u32) -> CmdStroke {
    let tile = ptcl[cmd_ix + 1u];
    let half_width = bitcast<f32>(ptcl[cmd_ix + 2u]);
    return CmdStroke(tile, half_width);
}

fn read_color(cmd_ix: u32) -> CmdColor {
    let rgba_color = ptcl[cmd_ix + 1u];
    return CmdColor(rgba_color);
}

fn read_lin_grad(cmd_ix: u32) -> CmdLinGrad {
    let index_mode = ptcl[cmd_ix + 1u];
    let index = index_mode >> 2u;
    let extend_mode = index_mode & 0x3u;
    let info_offset = ptcl[cmd_ix + 2u];
    let line_x = bitcast<f32>(info[info_offset]);
    let line_y = bitcast<f32>(info[info_offset + 1u]);
    let line_c = bitcast<f32>(info[info_offset + 2u]);
    return CmdLinGrad(index, extend_mode, line_x, line_y, line_c);
}

fn read_rad_grad(cmd_ix: u32) -> CmdRadGrad {
    let index_mode = ptcl[cmd_ix + 1u];
    let index = index_mode >> 2u;
    let extend_mode = index_mode & 0x3u;
    let info_offset = ptcl[cmd_ix + 2u];
    let matrx = bitcast<vec4<f32>>(vec4(
        info[info_offset],
        info[info_offset + 1u],
        info[info_offset + 2u],
        info[info_offset + 3u]
    ));
    let xlat = bitcast<vec2<f32>>(vec2(info[info_offset + 4u], info[info_offset + 5u]));
    let focal_x = bitcast<f32>(info[info_offset + 6u]);
    let radius = bitcast<f32>(info[info_offset + 7u]);
    let flags_kind = info[info_offset + 8u];
    let flags = flags_kind >> 3u;
    let kind = flags_kind & 0x7u;
    return CmdRadGrad(index, extend_mode, matrx, xlat, focal_x, radius, kind, flags);
}

fn read_image(cmd_ix: u32) -> CmdImage {
    let info_offset = ptcl[cmd_ix + 1u];
    let matrx = bitcast<vec4<f32>>(vec4(
        info[info_offset],
        info[info_offset + 1u],
        info[info_offset + 2u],
        info[info_offset + 3u]
    ));
    let xlat = bitcast<vec2<f32>>(vec2(info[info_offset + 4u], info[info_offset + 5u]));
    let xy = info[info_offset + 6u];
    let width_height = info[info_offset + 7u];
    let extend_mode = info[info_offset + 8u];
    // The following are not intended to be bitcasts
    let x = f32(xy >> 16u);
    let y = f32(xy & 0xffffu);
    let width = f32(width_height >> 16u);
    let height = f32(width_height & 0xffffu);
    let extend = vec2(extend_mode >> 2u, extend_mode & 0x3);
    return CmdImage(matrx, xlat, vec2(x, y), vec2(width, height), extend);
}

fn read_end_clip(cmd_ix: u32) -> CmdEndClip {
    let blend = ptcl[cmd_ix + 1u];
    let color_matrx_0 = bitcast<vec4<f32>>(vec4(ptcl[cmd_ix + 2u], ptcl[cmd_ix + 3u], ptcl[cmd_ix + 4u], ptcl[cmd_ix + 5u]));
    let color_matrx_1 = bitcast<vec4<f32>>(vec4(ptcl[cmd_ix + 6u], ptcl[cmd_ix + 7u], ptcl[cmd_ix + 8u], ptcl[cmd_ix + 9u]));
    let color_matrx_2 = bitcast<vec4<f32>>(vec4(ptcl[cmd_ix + 10u], ptcl[cmd_ix + 11u], ptcl[cmd_ix + 12u], ptcl[cmd_ix + 13u]));
    let color_matrx_3 = bitcast<vec4<f32>>(vec4(ptcl[cmd_ix + 14u], ptcl[cmd_ix + 15u], ptcl[cmd_ix + 16u], ptcl[cmd_ix + 17u]));
    let color_matrx_4 = bitcast<vec4<f32>>(vec4(ptcl[cmd_ix + 18u], ptcl[cmd_ix + 19u], ptcl[cmd_ix + 20u], ptcl[cmd_ix + 21u]));
    let needs_un_premultiply = ptcl[cmd_ix + 22u] == 1u;
    return CmdEndClip(blend, color_matrx_0, color_matrx_1, color_matrx_2, color_matrx_3, color_matrx_4, needs_un_premultiply);
}

let EXTEND_PAD = 0u;
let EXTEND_REPEAT = 1u;
let EXTEND_REFLECT = 2u;

fn extend_mode(t: f32, mode: u32) -> f32 {
    switch mode {
        case EXTEND_PAD: {
            return clamp(t, 0.0, 1.0);
        }
        case EXTEND_REPEAT: {
            return fract(t);
        }
        // EXTEND_REFLECT
        default: {
            return abs(t - 2.0 * round(0.5 * t));
        }
    }
}

#else

@group(0) @binding(3)
var output: texture_storage_2d<r8, write>;

#endif

let PIXELS_PER_THREAD = 4u;

fn fill_path(tile: Tile, xy: vec2<f32>, even_odd: bool) -> array<f32, PIXELS_PER_THREAD> {
    var area: array<f32, PIXELS_PER_THREAD>;
    let backdrop_f = f32(tile.backdrop);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        area[i] = backdrop_f;
    }
    var segment_ix = tile.segments;
    while segment_ix != 0u {
        let segment = segments[segment_ix];
        let y = segment.origin.y - xy.y;
        let y0 = clamp(y, 0.0, 1.0);
        let y1 = clamp(y + segment.delta.y, 0.0, 1.0);
        let dy = y0 - y1;
        if dy != 0.0 {
            let vec_y_recip = 1.0 / segment.delta.y;
            let t0 = (y0 - y) * vec_y_recip;
            let t1 = (y1 - y) * vec_y_recip;
            let startx = segment.origin.x - xy.x;
            let x0 = startx + t0 * segment.delta.x;
            let x1 = startx + t1 * segment.delta.x;
            let xmin0 = min(x0, x1);
            let xmax0 = max(x0, x1);
            for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                let i_f = f32(i);
                let xmin = min(xmin0 - i_f, 1.0) - 1.0e-6;
                let xmax = xmax0 - i_f;
                let b = min(xmax, 1.0);
                let c = max(b, 0.0);
                let d = max(xmin, 0.0);
                let a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
                area[i] += a * dy;
            }
        }
        let y_edge = sign(segment.delta.x) * clamp(xy.y - segment.y_edge + 1.0, 0.0, 1.0);
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            area[i] += y_edge;
        }
        segment_ix = segment.next;
    }
    if even_odd {
        // even-odd winding rule
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            let a = area[i];
            area[i] = abs(a - 2.0 * round(0.5 * a));
        }
    } else {
        // non-zero winding rule
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            area[i] = min(abs(area[i]), 1.0);
        }
    }
    return area;
}

fn stroke_path(seg: u32, half_width: f32, xy: vec2<f32>) -> array<f32, PIXELS_PER_THREAD> {
    var df: array<f32, PIXELS_PER_THREAD>;
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        df[i] = 1e9;
    }
    var segment_ix = seg;
    while segment_ix != 0u {
        let segment = segments[segment_ix];
        let delta = segment.delta;
        let dpos0 = xy + vec2(0.5, 0.5) - segment.origin;
        let scale = 1.0 / dot(delta, delta);
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            let dpos = vec2(dpos0.x + f32(i), dpos0.y);
            let t = clamp(dot(dpos, delta) * scale, 0.0, 1.0);
            // performance idea: hoist sqrt out of loop
            df[i] = min(df[i], length(delta * t - dpos));
        }
        segment_ix = segment.next;
    }
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        // reuse array; return alpha rather than distance
        df[i] = clamp(half_width + 0.5 - df[i], 0.0, 1.0);
    }
    return df;
}

// The X size should be 16 / PIXELS_PER_THREAD
@compute @workgroup_size(4, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let tile_ix = wg_id.y * config.width_in_tiles + wg_id.x;
    let xy = vec2(f32(global_id.x * PIXELS_PER_THREAD), f32(global_id.y));
#ifdef full
    var rgba: array<vec4<f32>, PIXELS_PER_THREAD>;
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        rgba[i] = unpack4x8unorm(config.base_color).wzyx;
    }
    var blend_stack: array<array<u32, PIXELS_PER_THREAD>, BLEND_STACK_SPLIT>;
    var clip_depth = 0u;
    var area: array<f32, PIXELS_PER_THREAD>;
    var cmd_ix = tile_ix * PTCL_INITIAL_ALLOC;
    let blend_offset = ptcl[cmd_ix];
    cmd_ix += 1u;
    // main interpretation loop
    while true {
        let tag = ptcl[cmd_ix];
        if tag == CMD_END {
            break;
        }
        switch tag {
            case CMD_FILL: {
                let fill = read_fill(cmd_ix);
                let segments = fill.tile >> 1u;
                let even_odd = (fill.tile & 1u) != 0u;
                let tile = Tile(fill.backdrop, segments);
                area = fill_path(tile, xy, even_odd);
                cmd_ix += 3u;
            }
            case CMD_STROKE: {
                let stroke = read_stroke(cmd_ix);
                area = stroke_path(stroke.tile, stroke.half_width, xy);
                cmd_ix += 3u;
            }
            case CMD_SOLID: {
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    area[i] = 1.0;
                }
                cmd_ix += 1u;
            }
            case CMD_COLOR: {
                let color = read_color(cmd_ix);
                let fg = unpack4x8unorm(color.rgba_color).wzyx;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let fg_i = fg * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 2u;
            }
            case CMD_LIN_GRAD: {
                let lin = read_lin_grad(cmd_ix);
                let d = lin.line_x * xy.x + lin.line_y * xy.y + lin.line_c;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_d = d + lin.line_x * f32(i);
                    let x = i32(round(extend_mode(my_d, lin.extend_mode) * f32(GRADIENT_WIDTH - 1)));
                    let fg_rgba = textureLoad(gradients, vec2(x, i32(lin.index)), 0);
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                cmd_ix += 3u;
            }
            case CMD_RAD_GRAD: {
                let rad = read_rad_grad(cmd_ix);
                let focal_x = rad.focal_x;
                let radius = rad.radius;
                let is_strip = rad.kind == RAD_GRAD_KIND_STRIP;
                let is_circular = rad.kind == RAD_GRAD_KIND_CIRCULAR;
                let is_focal_on_circle = rad.kind == RAD_GRAD_KIND_FOCAL_ON_CIRCLE;
                let is_swapped = (rad.flags & RAD_GRAD_SWAPPED) != 0u;
                let r1_recip = select(1.0 / radius, 0.0, is_circular);
                let less_scale = select(1.0, -1.0, is_swapped || (1.0 - focal_x) < 0.0);
                let t_sign = sign(1.0 - focal_x);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let local_xy = rad.matrx.xy * my_xy.x + rad.matrx.zw * my_xy.y + rad.xlat;
                    let x = local_xy.x;
                    let y = local_xy.y;
                    let xx = x * x;
                    let yy = y * y;
                    var t = 0.0;
                    var is_valid = true;
                    if is_strip {
                        let a = radius - yy;
                        t = sqrt(a) + x;
                        is_valid = a >= 0.0;
                    } else if is_focal_on_circle {
                        t = (xx + yy) / x;
                        is_valid = t >= 0.0 && x != 0.0;
                    } else if radius > 1.0 {
                        t = sqrt(xx + yy) - x * r1_recip;
                    } else { // radius < 1.0
                        let a = xx - yy;
                        t = less_scale * sqrt(a) - x * r1_recip;
                        is_valid = a >= 0.0 && t >= 0.0;
                    }
                    if is_valid {
                        t = extend_mode(focal_x + t_sign * t, rad.extend_mode);
                        t = select(t, 1.0 - t, is_swapped);
                        let x = i32(round(t * f32(GRADIENT_WIDTH - 1)));
                        let fg_rgba = textureLoad(gradients, vec2(x, i32(rad.index)), 0);
                        let fg_i = fg_rgba * area[i];
                        rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                    }
                }
                cmd_ix += 3u;
            }
            case CMD_IMAGE: {
                let image = read_image(cmd_ix);
                let atlas_extents = image.atlas_offset + image.extents;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let atlas_uv = image.matrx.xy * my_xy.x + image.matrx.zw * my_xy.y + image.xlat + image.atlas_offset;
                    if all(atlas_uv < atlas_extents) && area[i] != 0.0 {
                        let uv_quad = vec4<i32>(vec4(max(floor(atlas_uv), image.atlas_offset), min(ceil(atlas_uv), atlas_extents)));
                        let uv_frac = fract(atlas_uv);
                        let a = premul_alpha(textureLoad(image_atlas, uv_quad.xy, 0));
                        let b = premul_alpha(textureLoad(image_atlas, uv_quad.xw, 0));
                        let c = premul_alpha(textureLoad(image_atlas, uv_quad.zy, 0));
                        let d = premul_alpha(textureLoad(image_atlas, uv_quad.zw, 0));
                        let fg_rgba = mix(mix(a, b, uv_frac.y), mix(c, d, uv_frac.y), uv_frac.x);
                        let fg_i = fg_rgba * area[i];
                        rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                    }
                }
                cmd_ix += 2u;
            }
            case CMD_BEGIN_CLIP: {
                if clip_depth < BLEND_STACK_SPLIT {
                    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                        blend_stack[clip_depth][i] = pack4x8unorm(rgba[i]);
                        rgba[i] = vec4(0.0);
                    }
                } else {
                    // TODO: spill to memory
                }
                clip_depth += 1u;
                cmd_ix += 1u;
            }
            case CMD_END_CLIP: {
                let end_clip = read_end_clip(cmd_ix);

                clip_depth -= 1u;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    var bg_rgba: u32;
                    if clip_depth < BLEND_STACK_SPLIT {
                        bg_rgba = blend_stack[clip_depth][i];
                    } else {
                        // load from memory
                    }
                    let bg = unpack4x8unorm(bg_rgba);

                    var rgba_in = rgba[i];
                    var rgba_out: vec4<f32>;

                    // SVG filter spec (https://www.w3.org/TR/SVG11/filters.html#feColorMatrixElement) notes:
                    // | These matrices often perform an identity mapping in the alpha channel. If that is the case, an
                    // | implementation can avoid the costly undoing and redoing of the premultiplication for all pixels
                    // | with A = 1.

                    // The matrix multiplication is essentially:
                    // [ r1 ]   [ m00 m01 m02 m03 m04 ]   [ r0 ]
                    // [ g1 ]   [ m10 m11 m12 m13 m14 ]   [ g0 ]
                    // [ b1 ] = [ m20 m21 m22 m23 m24 ] * [ b0 ]
                    // [ a1 ]   [ m30 m31 m32 m33 m34 ]   [ a0 ]
                    //                                    [ 1 ]

                    // This condition should be the same for all invocations
                    if end_clip.needs_un_premultiply {
                        // Max with a small epsilon to avoid NaNs
                        let a_inv = 1.0 / max(rgba_in.a, 1e-6);
                        // Un-premultiply
                        rgba_in = vec4(rgba_in.rgb * a_inv, rgba_in.a);

                        // Homogeneous color-matrix multiply on the un-premultiplied color
                        rgba_out =
                            end_clip.color_matrx_0 * rgba_in.r +
                            end_clip.color_matrx_1 * rgba_in.g +
                            end_clip.color_matrx_2 * rgba_in.b +
                            end_clip.color_matrx_3 * rgba_in.a +
                            end_clip.color_matrx_4;

                        rgba_out = clamp(rgba_out, vec4(0.0), vec4(1.0));

                        // Premultiply again
                        rgba_out = vec4(rgba_out.rgb * rgba_out.a, rgba_out.a);
                    } else {
                        // Handling the case where output alpha (a1) is proportional to input alpha (a0), and does not
                        // depend on the RGB. This means m30=m31=m32=m34=0, and that a1=m33 * a0.
                        //
                        // Our matrix uses non-premultiplied alpha, so its input is (r0/a0, g0/a0, b0/a0, a0, 1)
                        // (it's homogeneous), and the output is (r1/a1, g1/a1, b1/a1, a1, 1), as column vectors
                        // (omitting the transpose notation)
                        //
                        // Thus for e.g. red:
                        // r1/a1 = (r0/a0)m00 + (g0/a0)m01 + (b0/a0)m02 + (a0)m03 + (1)m04
                        //
                        // with a1 = m33 * a0 and solving for r1, this becomes:
                        // r1 = m33 * ( (r0)m00 + (g0)m01 + (b0)m02 + (a0^2)m03 + (a0)m04 )
                        //
                        // thus:
                        // (r1, g1, b1) = m33 * M * (r0, g0, b0, a0^2, a0)
                        // a1 = m33 * a0
                        let new_rgb =
                            end_clip.color_matrx_0.rgb * rgba_in.r +
                            end_clip.color_matrx_1.rgb * rgba_in.g +
                            end_clip.color_matrx_2.rgb * rgba_in.b +
                            end_clip.color_matrx_3.rgb * rgba_in.a * rgba_in.a +
                            end_clip.color_matrx_4.rgb * rgba_in.a;
                        rgba_out = end_clip.color_matrx_3.a * vec4(new_rgb, rgba_in.a);

                        // Clamp down to ensure we're still validly premultiplied
                        rgba_out = clamp(rgba_out, vec4(0.0), vec4(min(1.0, rgba_out.a)));
                    }

                    let fg = rgba_out * area[i];
                    rgba[i] = blend_mix_compose(bg, fg, end_clip.blend);
                }
                cmd_ix += 23u;
            }
            case CMD_JUMP: {
                cmd_ix = ptcl[cmd_ix + 1u];
            }
            default: {}
        }
    }
    let xy_uint = vec2<u32>(xy);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let coords = xy_uint + vec2(i, 0u);
        if coords.x < config.target_width && coords.y < config.target_height {
            textureStore(output, vec2<i32>(coords), rgba[i]);
        }
    } 
#else
    let tile = tiles[tile_ix];
    let area = fill_path(tile, xy);

    let xy_uint = vec2<u32>(xy);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let coords = xy_uint + vec2(i, 0u);
        if coords.x < config.target_width && coords.y < config.target_height {
            textureStore(output, vec2<i32>(coords), vec4(area[i]));
        }
    }
#endif
}

fn premul_alpha(rgba: vec4<f32>) -> vec4<f32> {
    return vec4(rgba.rgb * rgba.a, rgba.a);
}
