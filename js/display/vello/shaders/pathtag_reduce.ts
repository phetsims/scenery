/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>u:a0;@group(0)@binding(2)var<storage,read_write>aQ:array<al>;const bZ=8u;const o=256u;var<workgroup>aG:array<al,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=u[n.dS+p];var agg=ew(K);aG[k.x]=agg;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=aG[k.x+(f<<i)];agg=bT(agg,au);}workgroupBarrier();aG[k.x]=agg;}if k.x==e{aQ[p>>bZ]=agg;}}`
