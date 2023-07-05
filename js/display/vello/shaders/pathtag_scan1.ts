/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<storage>aO:array<ak>;@group(0)@binding(1)var<storage>ho:array<ak>;@group(0)@binding(2)var<storage,read_write>b7:array<ak>;const bY=8u;const o=256u;var<workgroup>bo:array<ak,o>;var<workgroup>bX:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){var agg=g2();if k.x<ap.x{agg=ho[k.x];}bo[k.x]=agg;for(var j=e;j<bY;j+=f){workgroupBarrier();if k.x+(f<<j)<o{let au=bo[k.x+(f<<j)];agg=bT(agg,au);}workgroupBarrier();bo[k.x]=agg;}let p=N.x;agg=aO[p];bX[k.x]=agg;for(var j=e;j<bY;j+=f){workgroupBarrier();if k.x>=f<<j{let au=bX[k.x-(f<<j)];agg=bT(au,agg);}workgroupBarrier();bX[k.x]=agg;}workgroupBarrier();var T=bo[0];if k.x>e{T=bT(T,bX[k.x-f]);}b7[p]=T;}`
