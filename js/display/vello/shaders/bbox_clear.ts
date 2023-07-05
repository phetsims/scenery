/* eslint-disable */
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
@group(0)@binding(0)var<uniform>j:aM;struct c6{F:i,N:i,O:i,R:i,X:c,bI:d}@group(0)@binding(1)var<storage,read_write>bL:array<c6>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)J:I){let l=J.x;if l<j.iy{bL[l].F=0x7fffffff;bL[l].N=0x7fffffff;bL[l].O=-0x80000000;bL[l].R=-0x80000000;}}`
