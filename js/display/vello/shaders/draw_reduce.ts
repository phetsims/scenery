/* eslint-disable */
import util from './shared/util.js';
import drawtag from './shared/drawtag.js';
import config from './shared/config.js';

export default `${config}
${drawtag}
@group(0)@binding(0)var<uniform>f:aF;@group(0)@binding(1)var<storage>k:array<u32>;@group(0)@binding(2)var<storage,read_write>aD:array<aE>;const h=256u;var<workgroup>au:array<aE,h>;${util}
@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)E:vec3u,@builtin(local_invocation_id)d:vec3u,){let i=E.x;let D=gt(i);var agg=gP(D);au[d.x]=agg;for(var c=0u;c<firstTrailingBit(h);c+=1u){workgroupBarrier();if d.x+(1u<<c)<h{let ai=au[d.x+(1u<<c)];agg=ec(agg,ai);}workgroupBarrier();au[d.x]=agg;}if d.x==0u{aD[i>>firstTrailingBit(h)]=agg;}}`
