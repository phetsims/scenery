/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>bM:array<cJ>;@group(0)@binding(2)var<storage,read_write>X:array<bt>;const o=256u;var<workgroup>gE:array<j,o>;var<workgroup>cZ:array<j,o>;var<workgroup>gD:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let aG=N.x;var el=e;if aG<n.cN{let W=bM[aG];gE[k.x]=W.l.z-W.l.x;el=W.l.w-W.l.y;gD[k.x]=W.X;}cZ[k.x]=el;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){el+=cZ[k.x-(f<<i)];}workgroupBarrier();cZ[k.x]=el;}workgroupBarrier();let ip=cZ[o-f];for(var c_=k.x;c_<ip;c_+=o){var av=e;for(var i=e;i<firstTrailingBit(o);i+=f){let bE=av+((o/2u)>>i);if c_>=cZ[bE- f]{av=bE;}}let b0=gE[av];if b0>e{var ek=c_-select(e,cZ[av- f],av>e);var aJ=gD[av]+ek*b0;var a6=X[aJ].O;for(var x=f;x<b0;x+=f){aJ+=f;a6+=X[aJ].O;X[aJ].O=a6;}}}}`
