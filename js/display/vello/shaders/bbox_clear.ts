/* eslint-disable */
import config from './shared/config.js';

export default `${config}
@group(0)@binding(0)var<uniform>_m:_aG;struct _cf{x0:i32,y0:i32,x1:i32,y1:i32,_L:f32,_bz:u32}@group(0)@binding(1)var<storage,read_write>_bD:array<_cf>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)_G:vec3<u32>,){let ix=_G.x;if ix<_m._is{_bD[ix].x0=0x7fffffff;_bD[ix].y0=0x7fffffff;_bD[ix].x1=-0x80000000;_bD[ix].y1=-0x80000000;}}`
