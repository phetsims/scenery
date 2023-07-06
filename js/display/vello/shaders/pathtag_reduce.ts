/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>s:aX;@group(0)@binding(2)var<storage,read_write>aP:array<ak>;const bV=8u;const o=256u;var<workgroup>aG:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=s[n.dR+p];var agg=eu(K);aG[k.x]=agg;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let av=aG[k.x+(f<<i)];agg=bQ(agg,av);}workgroupBarrier();aG[k.x]=agg;}if k.x==e{aP[p>>bV]=agg;}}`
