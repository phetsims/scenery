/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${bbox}
${clip}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>cW:array<eB>;@group(0)@binding(2)var<storage>bK:array<c8>;@group(0)@binding(3)var<storage,read_write>aN:array<a3>;@group(0)@binding(4)var<storage,read_write>h8:array<fv>;const o=256u;var<workgroup>aM:array<a3,o>;var<workgroup>bo:array<j,o>;var<workgroup>gm:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let cX=cW[N.x].ac;let cb=cX>=0;var al=a3(f-j(cb),j(cb));aM[k.x]=al;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=aM[k.x+(f<<i)];al=dR(al,au);}workgroupBarrier();aM[k.x]=al;}if k.x==e{aN[ap.x]=al;}workgroupBarrier();let size=aM[0].b;al=a3();if cb&&al.a==e{let gn=size-al.b-f;bo[gn]=k.x;gm[gn]=j(cX);}workgroupBarrier();if k.x<size{let ac=gm[k.x];let aU=bK[ac];let dQ=bo[k.x]+ap.x*o;let l=vec4(h(aU.J),h(aU.S),h(aU.T),h(aU.V));h8[N.x]=fv(dQ,l);}}`
