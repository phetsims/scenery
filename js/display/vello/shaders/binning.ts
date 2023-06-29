/* eslint-disable */
import bump from './shared/bump.js';
import bbox from './shared/bbox.js';
import drawtag from './shared/drawtag.js';
import config from './shared/config.js';

export default `



${config}
${drawtag}
${bbox}
${bump}

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> draw_monoids: array<DrawMonoid>;

@group(0) @binding(2)
var<storage> path_bbox_buf: array<PathBbox>;

@group(0) @binding(3)
var<storage> clip_bbox_buf: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> intersected_bbox: array<vec4<f32>>;

@group(0) @binding(5)
var<storage, read_write> bump: BumpAllocators;

@group(0) @binding(6)
var<storage, read_write> bin_data: array<u32>;


struct BinHeader {
    element_count: u32,
    chunk_offset: u32,
}

@group(0) @binding(7)
var<storage, read_write> bin_header: array<BinHeader>;


const SX = 0.00390625;
const SY = 0.00390625;



const WG_SIZE = 256u;
const N_SLICE = 8u;

const N_SUBSLICE = 4u;

var<workgroup> sh_bitmaps: array<array<atomic<u32>, N_TILE>, N_SLICE>;

var<workgroup> sh_count: array<array<u32, N_TILE>, N_SUBSLICE>;
var<workgroup> sh_chunk_offset: array<u32, N_TILE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    for (var i = 0u; i < N_SLICE; i += 1u) {
        atomicStore(&sh_bitmaps[i][local_id.x], 0u);
    }
    workgroupBarrier();

    
    let element_ix = global_id.x;
    var x0 = 0;
    var y0 = 0;
    var x1 = 0;
    var y1 = 0;
    if element_ix < config.n_drawobj {
        let draw_monoid = draw_monoids[element_ix];
        var clip_bbox = vec4(-1e9, -1e9, 1e9, 1e9);
        if draw_monoid.clip_ix > 0u {
            
            
            
            clip_bbox = clip_bbox_buf[min(draw_monoid.clip_ix - 1u, config.n_clip - 1u)];
        }
        
        
        
        

        let path_bbox = path_bbox_buf[draw_monoid.path_ix];
        let pb = vec4<f32>(vec4(path_bbox.x0, path_bbox.y0, path_bbox.x1, path_bbox.y1));
        let bbox_raw = bbox_intersect(clip_bbox, pb);
        
        let bbox = vec4(bbox_raw.xy, max(bbox_raw.xy, bbox_raw.zw));

        intersected_bbox[element_ix] = bbox;
        x0 = i32(floor(bbox.x * SX));
        y0 = i32(floor(bbox.y * SY));
        x1 = i32(ceil(bbox.z * SX));
        y1 = i32(ceil(bbox.w * SY));
    }
    let width_in_bins = i32((config.width_in_tiles + N_TILE_X - 1u) / N_TILE_X);
    let height_in_bins = i32((config.height_in_tiles + N_TILE_Y - 1u) / N_TILE_Y);
    x0 = clamp(x0, 0, width_in_bins);
    y0 = clamp(y0, 0, height_in_bins);
    x1 = clamp(x1, 0, width_in_bins);
    y1 = clamp(y1, 0, height_in_bins);
    if x0 == x1 {
        y1 = y0;
    }
    var x = x0;
    var y = y0;
    let my_slice = local_id.x / 32u;
    let my_mask = 1u << (local_id.x & 31u);
    while y < y1 {
        atomicOr(&sh_bitmaps[my_slice][y * width_in_bins + x], my_mask);
        x += 1;
        if x == x1 {
            x = x0;
            y += 1;
        }
    }

    workgroupBarrier();
    
    var element_count = 0u;
    for (var i = 0u; i < N_SUBSLICE; i += 1u) {
        element_count += countOneBits(atomicLoad(&sh_bitmaps[i * 2u][local_id.x]));
        let element_count_lo = element_count;
        element_count += countOneBits(atomicLoad(&sh_bitmaps[i * 2u + 1u][local_id.x]));
        let element_count_hi = element_count;
        let element_count_packed = element_count_lo | (element_count_hi << 16u);
        sh_count[i][local_id.x] = element_count_packed;
    }
    
    var chunk_offset = atomicAdd(&bump.binning, element_count);
    if chunk_offset + element_count > config.binning_size {
        chunk_offset = 0u;
        atomicOr(&bump.failed, STAGE_BINNING);
    }    
    sh_chunk_offset[local_id.x] = chunk_offset;
    bin_header[global_id.x].element_count = element_count;
    bin_header[global_id.x].chunk_offset = chunk_offset;
    workgroupBarrier();

    
    x = x0;
    y = y0;
    while y < y1 {
        let bin_ix = y * width_in_bins + x;
        let out_mask = atomicLoad(&sh_bitmaps[my_slice][bin_ix]);
        
        if (out_mask & my_mask) != 0u {
            var idx = countOneBits(out_mask & (my_mask - 1u));
            if my_slice > 0u {
                let count_ix = my_slice - 1u;
                let count_packed = sh_count[count_ix / 2u][bin_ix];
                idx += (count_packed >> (16u * (count_ix & 1u))) & 0xffffu;
            }
            let offset = config.bin_data_start + sh_chunk_offset[bin_ix];
            bin_data[offset + idx] = element_ix;
        }
        x += 1;
        if x == x1 {
            x = x0;
            y += 1;
        }
    }
}
`
