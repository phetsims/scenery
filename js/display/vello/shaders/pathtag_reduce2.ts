/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<storage>hp:array<ak>;@group(0)@binding(1)var<storage,read_write>aN:array<ak>;const bU=8u;const o=256u;var<workgroup>aF:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;var agg=hp[p];aF[k.x]=agg;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=aF[k.x+(f<<i)];agg=bO(agg,au);}workgroupBarrier();aF[k.x]=agg;}if k.x==e{aN[p>>bU]=agg;}}`
