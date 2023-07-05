/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>j:aM;@group(0)@binding(1)var<storage>n:aS;@group(0)@binding(2)var<storage,read_write>aK:array<af>;const bU=8u;const k=256u;var<workgroup>aB:array<af,k>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)J:I,@builtin(local_invocation_id)f:I){let l=J.x;let H=n[j.dO+l];var agg=ei(H);aB[f.x]=agg;for(var e=0u;e<firstTrailingBit(k);e+=1u){workgroupBarrier();if f.x+(1u<<e)<k{let ao=aB[f.x+(1u<<e)];agg=bP(agg,ao);}workgroupBarrier();aB[f.x]=agg;}if f.x==0u{aK[l>>bU]=agg;}}`
