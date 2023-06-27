/* eslint-disable */
import config from './shared/config.js';

export default `${config}
@group(0)@binding(0)
var<uniform>_l:_aL;struct _bY{x0:i32,y0:i32,x1:i32,y1:i32,_I:f32,_bs:u32}@group(0)@binding(1)
var<storage,read_write>_bw:array<_bY>;@compute @workgroup_size(256)
fn main(
@builtin(global_invocation_id)_E:vec3<u32>,){let ix=_E.x;if ix<_l._il{_bw[ix].x0=0x7fffffff;_bw[ix].y0=0x7fffffff;_bw[ix].x1=-0x80000000;_bw[ix].y1=-0x80000000;}}`
