/* eslint-disable */
import util from './shared/util.js';
import drawtag from './shared/drawtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${drawtag}
@group(0)@binding(0)var<uniform>n:aT;@group(0)@binding(1)var<storage>u:aZ;@group(0)@binding(2)var<storage,read_write>aO:array<aS>;const o=256u;var<workgroup>aG:array<aS,o>;${util}
@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=gN(p);var agg=ha(K);aG[k.x]=agg;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=aG[k.x+(f<<i)];agg=eA(agg,au);}workgroupBarrier();aG[k.x]=agg;}if k.x==e{aO[p>>firstTrailingBit(o)]=agg;}}`
