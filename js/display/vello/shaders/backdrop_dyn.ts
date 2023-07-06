/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>bN:array<cJ>;@group(0)@binding(2)var<storage,read_write>X:array<bu>;const o=256u;var<workgroup>gG:array<j,o>;var<workgroup>cZ:array<j,o>;var<workgroup>gF:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let aG=N.x;var ek=e;if aG<n.cN{let W=bN[aG];gG[k.x]=W.l.z-W.l.x;ek=W.l.w-W.l.y;gF[k.x]=W.X;}cZ[k.x]=ek;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){ek+=cZ[k.x-(f<<i)];}workgroupBarrier();cZ[k.x]=ek;}workgroupBarrier();let it=cZ[o-f];for(var c_=k.x;c_<it;c_+=o){var av=e;for(var i=e;i<firstTrailingBit(o);i+=f){let bF=av+((o/2u)>>i);if c_>=cZ[bF- f]{av=bF;}}let b1=gG[av];if b1>e{var ej=c_-select(e,cZ[av- f],av>e);var aJ=gF[av]+ej*b1;var a8=X[aJ].O;for(var x=f;x<b1;x+=f){aJ+=f;a8+=X[aJ].O;X[aJ].O=a8;}}}}`
