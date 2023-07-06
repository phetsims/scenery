/* eslint-disable */
import tile from './shared/tile.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${tile}
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>bN:array<cJ>;@group(0)@binding(2)var<storage,read_write>W:array<bs>;const o=256u;var<workgroup>gG:array<j,o>;var<workgroup>cZ:array<j,o>;var<workgroup>gF:array<j,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let aH=N.x;var eo=e;if aH<n.cN{let X=bN[aH];gG[k.x]=X.l.z-X.l.x;eo=X.l.w-X.l.y;gF[k.x]=X.W;}cZ[k.x]=eo;for(var i=e;i<firstTrailingBit(o);i+=f){workgroupBarrier();if k.x>=(f<<i){eo+=cZ[k.x-(f<<i)];}workgroupBarrier();cZ[k.x]=eo;}workgroupBarrier();let ip=cZ[o-f];for(var c_=k.x;c_<ip;c_+=o){var aw=e;for(var i=e;i<firstTrailingBit(o);i+=f){let bE=aw+((o/2u)>>i);if c_>=cZ[bE- f]{aw=bE;}}let cg=gG[aw];if cg>e{var en=c_-select(e,cZ[aw- f],aw>e);var aK=gF[aw]+en*cg;var a6=W[aK].P;for(var x=f;x<cg;x+=f){aK+=f;a6+=W[aK].P;W[aK].P=a6;}}}}`
