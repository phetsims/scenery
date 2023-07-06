/* eslint-disable */
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
@group(0)@binding(0)var<uniform>n:aV;struct db{J:m,S:m,T:m,V:m,ad:h,bM:j}@group(0)@binding(1)var<storage,read_write>bP:array<db>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M){let p=N.x;if p<n.iG{bP[p].J=0x7fffffff;bP[p].S=0x7fffffff;bP[p].T=-0x80000000;bP[p].V=-0x80000000;}}`
