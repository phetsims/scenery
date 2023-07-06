/* eslint-disable */

export default `struct aR{P:L,b4:B}fn c2(Z:aR,bQ:B)->B{return Z.P.xy*bQ.x+Z.P.zw*bQ.y+Z.b4;}fn fq(Z:aR)->aR{let iv=d/(Z.P.x*Z.P.w-Z.P.y*Z.P.z);let fp=iv*vec4(Z.P.w,-Z.P.y,-Z.P.z,Z.P.x);let iu=mat2x2(fp.xy,fp.zw)*-Z.b4;return aR(fp,iu);}fn dK(a:aR,b:aR)->aR{return aR(a.P.xyxy*b.P.xxzz+a.P.zwzw*b.P.yyww,a.P.xy*b.b4.x+a.P.zw*b.b4.y+a.b4);}fn gO(eB:j,p:j)->aR{let aQ=eB+p*6u;let P=bitcast<L>(vec4(u[aQ],u[aQ+f],u[aQ+2u],u[aQ+3u]));let b4=bitcast<B>(vec2(u[aQ+4u],u[aQ+5u]));return aR(P,b4);}`
