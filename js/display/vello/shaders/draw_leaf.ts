/* eslint-disable */
import util from './shared/util.js';
import transform from './shared/transform.js';
import bbox from './shared/bbox.js';
import drawtag from './shared/drawtag.js';
import clip from './shared/clip.js';
import config from './shared/config.js';
import pre from './shared/pre.js';

export default `${pre}${config}
${clip}
${drawtag}
${bbox}
${transform}
@group(0)@binding(0)var<uniform>n:aQ;@group(0)@binding(1)var<storage>s:aW;@group(0)@binding(2)var<storage>aO:array<aP>;@group(0)@binding(3)var<storage>aT:array<da>;@group(0)@binding(4)var<storage,read_write>dK:array<aP>;@group(0)@binding(5)var<storage,read_write>info:aW;@group(0)@binding(6)var<storage,read_write>cX:array<eu>;${util}
const o=256u;fn fa(c6:i,p:i)->aK{let am=c6+p*6u;let e9=bitcast<h>(s[am]);let e8=bitcast<h>(s[am+f]);let e7=bitcast<h>(s[am+2u]);let e6=bitcast<h>(s[am+3u]);let e5=bitcast<h>(s[am+4u]);let e4=bitcast<h>(s[am+5u]);let K=vec4(e9,e8,e7,e6);let be=vec2(e5,e4);return aK(K,be);}var<workgroup>aF:array<aP,o>;@compute @workgroup_size(256)fn main(@builtin(global_invocation_id)N:M,@builtin(local_invocation_id)k:M,@builtin(workgroup_id)ap:M){let p=N.x;var agg=iv();if k.x<ap.x{agg=aO[k.x];}aF[k.x]=agg;for(var j=e;j<firstTrailingBit(o);j+=f){workgroupBarrier();if k.x+(f<<j)<o{let au=aF[k.x+(f<<j)];agg=er(agg,au);}workgroupBarrier();aF[k.x]=agg;}workgroupBarrier();var bw=aF[0];workgroupBarrier();let L=gI(p);agg=g3(L);aF[k.x]=agg;for(var j=e;j<firstTrailingBit(o);j+=f){workgroupBarrier();if k.x>=f<<j{let au=aF[k.x-(f<<j)];agg=er(agg,au);}workgroupBarrier();aF[k.x]=agg;}workgroupBarrier();if k.x>e{bw=er(bw,aF[k.x- f]);}if p<n.cN{dK[p]=bw;}let E=n.fy+bw.bK;let Z=bw.Q;if L==fw||L==eq||L==ep||L==eo||L==dR{let l=aT[bw.ac];let ji=i(l.ad>=c);var Y=aK();var ad=l.ad;if ad>=c||L==eq||L==ep||L==eo{Y=fa(n.c6,l.bM);}if ad>=c{let K=Y.K;ad*=sqrt(abs(K.x*K.w-K.y*K.z));}switch L{case 66u:{info[Z]=bitcast<i>(ad);}case 266u:{info[Z]=bitcast<i>(ad);var t=bitcast<B>(vec2(s[E+f],s[E+2u]));var v=bitcast<B>(vec2(s[E+3u],s[E+4u]));t=c1(Y,t);v=c1(Y,v);let e3=v-t;let a9=d/dot(e3,e3);let e2=e3*a9;let dN=-dot(t,e2);info[Z+f]=bitcast<i>(e2.x);info[Z+2u]=bitcast<i>(e2.y);info[Z+3u]=bitcast<i>(dN);}case 654u:{let e1=d/h(1<<12u);info[Z]=bitcast<i>(ad);var t=bitcast<B>(vec2(s[E+f],s[E+2u]));var v=bitcast<B>(vec2(s[E+3u],s[E+4u]));var cT=bitcast<h>(s[E+5u]);var cE=bitcast<h>(s[E+6u]);let f8=fp(Y);var cc=aK();var a6=c;var aU=c;var bJ=e;var aE=e;if abs(cT-cE)<=e1{bJ=g9;let f7=cT/distance(t,v);cc=dL(gc(t,v),f8);aU=f7*f7;}else{bJ=iE;if all(t==v){bJ=ha;t+=e1;}if cE==c{aE|=g7;let hU=t;t=v;v=hU;let hT=cT;cT=cE;cE=hT;}a6=cT/(cT-cE);let f6=(d-a6)*t+a6*v;aU=cE/(distance(f6,v));let e0=dL(gc(f6,v),f8);var e_=e0;if abs(aU- d)<=e1{bJ=g8;let a9=.5*abs(d-a6);e_=dL(aK(vec4(a9,c,0.,a9),vec2(c)),e0);}else{let a=aU*aU- d;let f5=abs(d-a6)/a;let hS=aU*f5;let hR=sqrt(abs(a))*f5;e_=dL(aK(vec4(hS,c,0.,hR),vec2(c)),e0);}cc=e_;}info[Z+f]=bitcast<i>(cc.K.x);info[Z+2u]=bitcast<i>(cc.K.y);info[Z+3u]=bitcast<i>(cc.K.z);info[Z+4u]=bitcast<i>(cc.K.w);info[Z+5u]=bitcast<i>(cc.be.x);info[Z+6u]=bitcast<i>(cc.be.y);info[Z+7u]=bitcast<i>(a6);info[Z+8u]=bitcast<i>(aU);info[Z+9u]=bitcast<i>((aE<<3u)|bJ);}case 580u:{info[Z]=bitcast<i>(ad);let cd=fp(Y);info[Z+f]=bitcast<i>(cd.K.x);info[Z+2u]=bitcast<i>(cd.K.y);info[Z+3u]=bitcast<i>(cd.K.z);info[Z+4u]=bitcast<i>(cd.K.w);info[Z+5u]=bitcast<i>(cd.be.x);info[Z+6u]=bitcast<i>(cd.be.y);info[Z+7u]=s[E];info[Z+8u]=s[E+f];}default:{}}}if L==dR||L==en{var ac=~p;if L==dR{ac=bw.ac;}cX[bw.cl]=eu(p,m(ac));}}fn gc(t:B,v:B)->aK{let hQ=f9(t,v);let cd=fp(hQ);let hP=f9(vec2(c),vec2(d,c));return dL(hP,cd);}fn f9(t:B,v:B)->aK{return aK(vec4(v.y-t.y,t.x-v.x,v.x-t.x,v.y-t.y),vec2(t.x,t.y));}`