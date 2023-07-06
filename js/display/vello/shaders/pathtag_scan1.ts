/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<storage>aP:array<ak>;@group(0)@binding(1)var<storage>ho:array<ak>;@group(0)@binding(2)var<storage,read_write>b6:array<ak>;const bV=8u;const o=256u;var<workgroup>bo:array<ak,o>;var<workgroup>bU:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)aq:M){var agg=g2();if k.x<aq.x{agg=ho[k.x];}bo[k.x]=agg;for(var i=e;i<bV;i+=f){workgroupBarrier();if k.x+(f<<i)<o{let av=bo[k.x+(f<<i)];agg=bQ(agg,av);}workgroupBarrier();bo[k.x]=agg;}let p=N.x;agg=aP[p];bU[k.x]=agg;for(var i=e;i<bV;i+=f){workgroupBarrier();if k.x>=f<<i{let av=bU[k.x-(f<<i)];agg=bQ(av,agg);}workgroupBarrier();bU[k.x]=agg;}workgroupBarrier();var U=bo[0];if k.x>e{U=bQ(U,bU[k.x-f]);}b6[p]=U;}`
