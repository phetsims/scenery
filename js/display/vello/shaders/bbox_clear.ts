/* eslint-disable */
import config from './shared/config.js';

export default `${config}
@group(0)@binding(0)var<uniform>f:aF;struct cY{B:i32,I:i32,J:i32,M:i32,S:f32,bA:u32}@group(0)@binding(1)var<storage,read_write>bD:array<cY>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)E:vec3u,){let i=E.x;if i<f.in{bD[i].B=0x7fffffff;bD[i].I=0x7fffffff;bD[i].J=-0x80000000;bD[i].M=-0x80000000;}}`
