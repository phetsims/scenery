/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `




${config}
${pathtag}

@group(0) @binding(0)
var<storage> reduced: array<TagMonoid>;

@group(0) @binding(1)
var<storage> reduced2: array<TagMonoid>;

@group(0) @binding(2)
var<storage, read_write> tag_monoids: array<TagMonoid>;

const LG_WG_SIZE = 8u;
const WG_SIZE = 256u;

var<workgroup> sh_parent: array<TagMonoid, WG_SIZE>;

var<workgroup> sh_monoid: array<TagMonoid, WG_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    var agg = tag_monoid_identity();
    if local_id.x < wg_id.x {
        agg = reduced2[local_id.x];
    }
    sh_parent[local_id.x] = agg;
    for (var i = 0u; i < LG_WG_SIZE; i += 1u) {
        workgroupBarrier();
        if local_id.x + (1u << i) < WG_SIZE {
            let other = sh_parent[local_id.x + (1u << i)];
            agg = combine_tag_monoid(agg, other);
        }
        workgroupBarrier();
        sh_parent[local_id.x] = agg;
    }

    let ix = global_id.x;
    agg = reduced[ix];
    sh_monoid[local_id.x] = agg;
    for (var i = 0u; i < LG_WG_SIZE; i += 1u) {
        workgroupBarrier();
        if local_id.x >= 1u << i {
            let other = sh_monoid[local_id.x - (1u << i)];
            agg = combine_tag_monoid(other, agg);
        }
        workgroupBarrier();
        sh_monoid[local_id.x] = agg;
    }
    workgroupBarrier();
    
    var tm = sh_parent[0];
    if local_id.x > 0u {
        tm = combine_tag_monoid(tm, sh_monoid[local_id.x - 1u]);
    }
    
    tag_monoids[ix] = tm;
}
`
