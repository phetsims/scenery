/* eslint-disable */
import util from './shared/util.js';
import drawtag from './shared/drawtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${drawtag}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>s:aX;@group(0)@binding(2)var<storage,read_write>aP:array<aQ>;const o=256u;var<workgroup>aG:array<aQ,o>;${util}
@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let L=gI(p);var agg=g3(L);aG[k.x]=agg;for(var j=e;j<firstTrailingBit(o);j+=f){workgroupBarrier();if k.x+(f<<j)<o{let aw=aG[k.x+(f<<j)];agg=er(agg,aw);}workgroupBarrier();aG[k.x]=agg;}if k.x==e{aP[p>>firstTrailingBit(o)]=agg;}}`
