/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<storage>aP:array<al>;@group(0)@binding(1)var<storage>ho:array<al>;@group(0)@binding(2)var<storage,read_write>ca:array<al>;const b_=8u;const o=256u;var<workgroup>bp:array<al,o>;var<workgroup>bZ:array<al,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)at:M){var agg=g2();if k.x<at.x{agg=ho[k.x];}bp[k.x]=agg;for(var j=e;j<b_;j+=f){workgroupBarrier();if k.x+(f<<j)<o{let aw=bp[k.x+(f<<j)];agg=bV(agg,aw);}workgroupBarrier();bp[k.x]=agg;}let p=N.x;agg=aP[p];bZ[k.x]=agg;for(var j=e;j<b_;j+=f){workgroupBarrier();if k.x>=f<<j{let aw=bZ[k.x-(f<<j)];agg=bV(aw,agg);}workgroupBarrier();bZ[k.x]=agg;}workgroupBarrier();var U=bp[0];if k.x>e{U=bV(U,bZ[k.x-f]);}ca[p]=U;}`
