/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<storage>hl:array<af>;@group(0)@binding(1)var<storage,read_write>aK:array<af>;const bU=8u;const k=256u;var<workgroup>aB:array<af,k>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)J:I,@builtin(local_invocation_id)f:I){let l=J.x;var agg=hl[l];aB[f.x]=agg;for(var e=0u;e<firstTrailingBit(k);e+=1u){workgroupBarrier();if f.x+(1u<<e)<k{let ao=aB[f.x+(1u<<e)];agg=bP(agg,ao);}workgroupBarrier();aB[f.x]=agg;}if f.x==0u{aK[l>>bU]=agg;}}`
