/* eslint-disable */
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
@group(0)@binding(0)var<storage>aN:array<ak>;@group(0)@binding(1)var<storage>ho:array<ak>;@group(0)@binding(2)var<storage,read_write>b6:array<ak>;const bU=8u;const o=256u;var<workgroup>bo:array<ak,o>;var<workgroup>bT:array<ak,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){var agg=g2();if k.x<ap.x{agg=ho[k.x];}bo[k.x]=agg;for(var i=e;i<bU;i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=bo[k.x+(f<<i)];agg=bP(agg,au);}workgroupBarrier();bo[k.x]=agg;}let p=N.x;agg=aN[p];bT[k.x]=agg;for(var i=e;i<bU;i+=f){workgroupBarrier();if k.x>=f<<i{let au=bT[k.x-(f<<i)];agg=bP(au,agg);}workgroupBarrier();bT[k.x]=agg;}workgroupBarrier();var U=bo[0];if k.x>e{U=bP(U,bT[k.x-f]);}b6[p]=U;}`
