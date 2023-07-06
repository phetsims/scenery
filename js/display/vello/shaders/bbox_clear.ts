/* eslint-disable */
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
@group(0)@binding(0)var<uniform>n:aR;struct c8{J:m,S:m,T:m,V:m,ad:h,bH:j}@group(0)@binding(1)var<storage,read_write>bK:array<c8>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M){let p=N.x;if p<n.iC{bK[p].J=0x7fffffff;bK[p].S=0x7fffffff;bK[p].T=-0x80000000;bK[p].V=-0x80000000;}}`
