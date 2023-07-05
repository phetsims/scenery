/* eslint-disable */
import util from './shared/util.js';
import drawtag from './shared/drawtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${drawtag}
@group(0)@binding(0)var<uniform>j:aM;@group(0)@binding(1)var<storage>n:aS;@group(0)@binding(2)var<storage,read_write>aK:array<aL>;const k=256u;var<workgroup>aB:array<aL,k>;${util}
@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)J:I,@builtin(local_invocation_id)f:I){let l=J.x;let H=gE(l);var agg=g_(H);aB[f.x]=agg;for(var e=0u;e<firstTrailingBit(k);e+=1u){workgroupBarrier();if f.x+(1u<<e)<k{let ao=aB[f.x+(1u<<e)];agg=en(agg,ao);}workgroupBarrier();aB[f.x]=agg;}if f.x==0u{aK[l>>firstTrailingBit(k)]=agg;}}`
