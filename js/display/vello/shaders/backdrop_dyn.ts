/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>bM:array<cJ>;@group(0)@binding(2)var<storage,read_write>X:array<bt>;const o=256u;var<workgroup>gG:array<j,o>;var<workgroup>cZ:array<j,o>;var<workgroup>gF:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let aG=N.x;var em=e;if aG<n.cN{let W=bM[aG];gG[k.x]=W.l.z-W.l.x;em=W.l.w-W.l.y;gF[k.x]=W.X;}cZ[k.x]=em;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){em+=cZ[k.x-(f<<i)];}workgroupBarrier();cZ[k.x]=em;}workgroupBarrier();let iq=cZ[o-f];for(var c_=k.x;c_<iq;c_+=o){var av=e;for(var i=e;i<firstTrailingBit(o);i+=f){let bE=av+((o/2u)>>i);if c_>=cZ[bE- f]{av=bE;}}let b0=gG[av];if b0>e{var el=c_-select(e,cZ[av- f],av>e);var aJ=gF[av]+el*b0;var a6=X[aJ].O;for(var x=f;x<b0;x+=f){aJ+=f;a6+=X[aJ].O;X[aJ].O=a6;}}}}`
