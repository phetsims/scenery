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
@group(0)@binding(0)var<uniform>n:aR;@group(0)@binding(1)var<storage>s:aX;@group(0)@binding(2)var<storage>ca:array<al>;struct hq{J:db,R:db,T:db,V:db,ad:h,bO:i}@group(0)@binding(3)var<storage,read_write>bR:array<hq>;@group(0)@binding(4)var<storage,read_write>eA:array<fx>;var<private>bv:i;fn b5(p:i)->B{let x=bitcast<h>(s[bv+p]);let y=bitcast<h>(s[bv+p+f]);return vec2(x,y);}fn b4(p:i)->B{let dv=s[bv+p];let x=h(m(dv<<16u)>>16u);let y=h(m(dv)>>16u);return vec2(x,y);}fn fa(c6:i,p:i)->aL{let an=c6+p*6u;let e9=bitcast<h>(s[an]);let e8=bitcast<h>(s[an+f]);let e7=bitcast<h>(s[an+2u]);let e6=bitcast<h>(s[an+3u]);let e5=bitcast<h>(s[an+4u]);let e4=bitcast<h>(s[an+5u]);let K=vec4(e9,e8,e7,e6);let bh=vec2(e5,e4);return aL(K,bh);}fn fO(x:h)->m{return m(floor(x));}fn fN(x:h)->m{return m(ceil(x));}@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M){let p=N.x;let L=s[n.dS+(p>>2u)];bv=n.bv;let cc=(p&3u)*8u;var U=em(L&((f<<cc)-f));U=bV(ca[p>>2u],U);var cb=(L>>cc)&255u;let bP=&bR[U.ac];let ad=bitcast<h>(s[n.iB+U.dO]);if(cb&g_)!=e{(*bP).ad=ad;(*bP).bO=U.bO;}let aS=cb&fv;if aS!=e{var t:B;var v:B;var H:B;var P:B;if(cb&g0)!=e{t=b5(U.aI);v=b5(U.aI+2u);if aS>=c5{H=b5(U.aI+4u);if aS==dQ{P=b5(U.aI+6u);}}}else{t=b4(U.aI);v=b4(U.aI+f);if aS>=c5{H=b4(U.aI+2u);if aS==dQ{P=b4(U.aI+3u);}}}let Y=fa(n.c6,U.bO);t=c1(Y,t);v=c1(Y,v);var l=vec4(min(t,v),max(t,v));if aS==g1{P=v;H=mix(P,t,d/3.);v=mix(t,P,d/3.);}else if aS>=c5{H=c1(Y,H);l=vec4(min(l.xy,H),max(l.zw,H));if aS==dQ{P=c1(Y,P);l=vec4(min(l.xy,P),max(l.zw,P));}else{P=H;H=mix(v,H,d/3.);v=mix(v,t,d/3.);}}var a2=vec2(c,0.);if ad>=c{a2=.5*ad*vec2(length(Y.K.xz),length(Y.K.yw));l+=vec4(-a2,a2);}let aF=i(ad>=c);eA[N.x]=fx(t,v,H,P,a2,U.ac,aF);if l.z>l.x||l.w>l.y{atomicMin(&(*bP).J,fO(l.x));atomicMin(&(*bP).R,fO(l.y));atomicMax(&(*bP).T,fN(l.z));atomicMax(&(*bP).V,fN(l.w));}}}`
