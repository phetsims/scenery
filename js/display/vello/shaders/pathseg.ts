/* eslint-disable */
import transform from './shared/transform.js';
import cubic from './shared/cubic.js';
import pathtag from './shared/pathtag.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${pathtag}
${cubic}
${transform}
@group(0)@binding(0)var<uniform>j:aM;@group(0)@binding(1)var<storage>n:aS;@group(0)@binding(2)var<storage>b3:array<af>;struct hm{F:c7,N:c7,O:c7,R:c7,X:c,bI:d}@group(0)@binding(3)var<storage,read_write>bL:array<hm>;@group(0)@binding(4)var<storage,read_write>ew:array<ft>;var<private>bp:d;fn b_(l:d)->t{let x=bitcast<c>(n[bp+l]);let y=bitcast<c>(n[bp+l+1u]);return vec2(x,y);}fn bZ(l:d)->t{let dr=n[bp+l];let x=c(i(dr<<16u)>>16u);let y=c(i(dr)>>16u);return vec2(x,y);}fn e6(c2:d,l:d)->aG{let ai=c2+l*6u;let e5=bitcast<c>(n[ai]);let e4=bitcast<c>(n[ai+1u]);let e3=bitcast<c>(n[ai+2u]);let e2=bitcast<c>(n[ai+3u]);let e1=bitcast<c>(n[ai+4u]);let e0=bitcast<c>(n[ai+5u]);let G=vec4(e5,e4,e3,e2);let a8=vec2(e1,e0);return aG(G,a8);}fn fK(x:c)->i{return i(floor(x));}fn fJ(x:c)->i{return i(ceil(x));}@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)J:I,@builtin(local_invocation_id)f:I){let l=J.x;let H=n[j.dO+(l>>2u)];bp=j.bp;let b5=(l&3u)*8u;var P=ei(H&((1u<<b5)- 1u));P=bP(b3[l>>2u],P);var b4=(H>>b5)&255u;let bJ=&bL[P.W];let X=bitcast<c>(n[j.ix+P.dK]);if(b4&gW)!=0u{(*bJ).X=X;(*bJ).bI=P.bI;}let aN=b4&fr;if aN!=0u{var o:t;var q:t;var D:t;var L:t;if(b4&gX)!=0u{o=b_(P.aD);q=b_(P.aD+2u);if aN>=c1{D=b_(P.aD+4u);if aN==dM{L=b_(P.aD+6u);}}}else{o=bZ(P.aD);q=bZ(P.aD+1u);if aN>=c1{D=bZ(P.aD+2u);if aN==dM{L=bZ(P.aD+3u);}}}let U=e6(j.c2,P.bI);o=cY(U,o);q=cY(U,q);var h=vec4(min(o,q),max(o,q));if aN==gY{L=q;D=mix(L,o,1./3.);q=mix(o,L,1./3.);}else if aN>=c1{D=cY(U,D);h=vec4(min(h.xy,D),max(h.zw,D));if aN==dM{L=cY(U,L);h=vec4(min(h.xy,L),max(h.zw,L));}else{L=D;D=mix(q,D,1./3.);q=mix(q,o,1./3.);}}var aW=vec2(0.,0.);if X>=0.{aW=.5*X*vec2(length(U.G.xz),length(U.G.yw));h+=vec4(-aW,aW);}let aA=d(X>=0.);ew[J.x]=ft(o,q,D,L,aW,P.W,aA);if h.z>h.x||h.w>h.y{atomicMin(&(*bJ).F,fK(h.x));atomicMin(&(*bJ).N,fK(h.y));atomicMax(&(*bJ).O,fJ(h.z));atomicMax(&(*bJ).R,fJ(h.w));}}}`
