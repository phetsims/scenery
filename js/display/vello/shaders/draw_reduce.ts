/* eslint-disable */
import util from './shared/util.js';
import drawtag from './shared/drawtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${drawtag}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>u:aX;@group(0)@binding(2)var<storage,read_write>aN:array<aQ>;const o=256u;var<workgroup>aF:array<aQ,o>;${util}
@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let K=gG(p);var agg=g4(K);aF[k.x]=agg;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=aF[k.x+(f<<i)];agg=ew(agg,au);}workgroupBarrier();aF[k.x]=agg;}if k.x==e{aN[p>>firstTrailingBit(o)]=agg;}}`
