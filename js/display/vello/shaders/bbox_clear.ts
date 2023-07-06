/* eslint-disable */
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
@group(0)@binding(0)var<uniform>n:aR;struct c8{J:m,S:m,T:m,V:m,ad:h,bI:j}@group(0)@binding(1)var<storage,read_write>bL:array<c8>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M){let p=N.x;if p<n.iC{bL[p].J=0x7fffffff;bL[p].S=0x7fffffff;bL[p].T=-0x80000000;bL[p].V=-0x80000000;}}`
