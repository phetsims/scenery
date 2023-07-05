/* eslint-disable */
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
@group(0)@binding(0)var<uniform>n:aR;struct da{J:m,R:m,T:m,V:m,ad:h,bO:i}@group(0)@binding(1)var<storage,read_write>bR:array<da>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M){let p=N.x;if p<n.iC{bR[p].J=0x7fffffff;bR[p].R=0x7fffffff;bR[p].T=-0x80000000;bR[p].V=-0x80000000;}}`
