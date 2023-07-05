/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
@group(0)@binding(0)var<storage>ha:array<X>;@group(0)@binding(1)var<storage,read_write>aD:array<X>;const bL=8u;const h=256u;var<workgroup>au:array<X,h>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)E:vec3u,@builtin(local_invocation_id)d:vec3u,){let i=E.x;var agg=ha[i];au[d.x]=agg;for(var c=0u;c<firstTrailingBit(h);c+=1u){workgroupBarrier();if d.x+(1u<<c)<h{let ai=au[d.x+(1u<<c)];agg=bH(agg,ai);}workgroupBarrier();au[d.x]=agg;}if d.x==0u{aD[i>>bL]=agg;}}`
