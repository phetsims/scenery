/* eslint-disable */
import clip from './shared/clip.js';
import bbox from './shared/bbox.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${bbox}
${clip}
@group(0)@binding(0)var<uniform>n:aT;@group(0)@binding(1)var<storage>cY:array<eE>;@group(0)@binding(2)var<storage>bO:array<da>;@group(0)@binding(3)var<storage,read_write>aO:array<a6>;@group(0)@binding(4)var<storage,read_write>ic:array<fA>;const o=256u;var<workgroup>aN:array<a6,o>;var<workgroup>bs:array<j,o>;var<workgroup>gu:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let cZ=cY[N.x].ac;let cd=cZ>=0;var am=a6(f-j(cd),j(cd));aN[k.x]=am;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x+(f<<i)<o{let au=aN[k.x+(f<<i)];am=dT(am,au);}workgroupBarrier();aN[k.x]=am;}if k.x==e{aO[ap.x]=am;}workgroupBarrier();let size=aN[0].b;am=a6();if cd&&am.a==e{let gv=size-am.b-f;bs[gv]=k.x;gu[gv]=j(cZ);}workgroupBarrier();if k.x<size{let ac=gu[k.x];let aW=bO[ac];let dS=bs[k.x]+ap.x*o;let l=vec4(h(aW.J),h(aW.S),h(aW.T),h(aW.V));ic[N.x]=fA(dS,l);}}`
