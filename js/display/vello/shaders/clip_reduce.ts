/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${bbox}
${clip}
@group(0)@binding(0)var<uniform>n:aV;@group(0)@binding(1)var<storage>cY:array<eE>;@group(0)@binding(2)var<storage>bO:array<da>;@group(0)@binding(3)var<storage,read_write>aQ:array<a7>;@group(0)@binding(4)var<storage,read_write>ic:array<fB>;const o=256u;var<workgroup>aP:array<a7,o>;var<workgroup>bs:array<j,o>;var<workgroup>gv:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let cZ=cY[N.x].ac;let cd=cZ>=0;var am=a7(f-j(cd),j(cd));aP[k.x]=am;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=aP[k.x+(f<<i)];am=dT(am,au);}workgroupBarrier();aP[k.x]=am;}if k.x==e{aQ[ap.x]=am;}workgroupBarrier();let size=aP[0].b;am=a7();if cd&&am.a==e{let gw=size-am.b-f;bs[gw]=k.x;gv[gw]=j(cZ);}workgroupBarrier();if k.x<size{let ac=gv[k.x];let aY=bO[ac];let dS=bs[k.x]+ap.x*o;let l=vec4(h(aY.J),h(aY.S),h(aY.T),h(aY.V));ic[N.x]=fB(dS,l);}}`
