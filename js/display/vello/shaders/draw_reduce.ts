/* eslint-disable */
import util from './shared/util.js';
import drawtag from './shared/drawtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${drawtag}
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>u:a0;@group(0)@binding(2)var<storage,read_write>aQ:array<aU>;const o=256u;var<workgroup>aG:array<aU,o>;${util}
@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=gO(p);var agg=hb(K);aG[k.x]=agg;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=aG[k.x+(f<<i)];agg=eA(agg,au);}workgroupBarrier();aG[k.x]=agg;}if k.x==e{aQ[p>>firstTrailingBit(o)]=agg;}}`
