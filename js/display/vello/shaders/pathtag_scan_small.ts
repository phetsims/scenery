/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';

export default `${config}
${pathtag}
@group(0)@binding(0)var<uniform>f:aF;@group(0)@binding(1)var<storage>k:array<u32>;@group(0)@binding(2)var<storage>aD:array<X>;@group(0)@binding(3)var<storage,read_write>bV:array<X>;const bL=8u;const h=256u;var<workgroup>a9:array<X,h>;var<workgroup>bK:array<X,h>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)E:vec3u,@builtin(local_invocation_id)d:vec3u,@builtin(workgroup_id)ae:vec3u,){var agg=gO();if d.x<ae.x{agg=aD[d.x];}a9[d.x]=agg;for(var c=0u;c<bL;c+=1u){workgroupBarrier();if d.x+(1u<<c)<h{let ai=a9[d.x+(1u<<c)];agg=bH(agg,ai);}workgroupBarrier();a9[d.x]=agg;}let i=E.x;let D=k[f.dD+i];var dH=d7(D);bK[d.x]=dH;for(var c=0u;c<bL;c+=1u){workgroupBarrier();if d.x>=1u<<c{let ai=bK[d.x-(1u<<c)];dH=bH(ai,dH);}workgroupBarrier();bK[d.x]=dH;}workgroupBarrier();var K=a9[0];if d.x>0u{K=bH(K,bK[d.x- 1u]);}bV[i]=K;}`
