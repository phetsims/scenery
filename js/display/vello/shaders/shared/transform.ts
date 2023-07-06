/* eslint-disable */

export default `struct aT{P:L,b4:B}fn c2(Z:aT,bQ:B)->B{return Z.P.xy*bQ.x+Z.P.zw*bQ.y+Z.b4;}fn fr(Z:aT)->aT{let iv=d/(Z.P.x*Z.P.w-Z.P.y*Z.P.z);let fq=iv*vec4(Z.P.w,-Z.P.y,-Z.P.z,Z.P.x);let iu=mat2x2(fq.xy,fq.zw)*-Z.b4;return aT(fq,iu);}fn dK(a:aT,b:aT)->aT{return aT(a.P.xyxy*b.P.xxzz+a.P.zwzw*b.P.yyww,a.P.xy*b.b4.x+a.P.zw*b.b4.y+a.b4);}fn gP(eB:j,p:j)->aT{let aS=eB+p*6u;let P=bitcast<L>(vec4(u[aS],u[aS+f],u[aS+2u],u[aS+3u]));let b4=bitcast<B>(vec2(u[aS+4u],u[aS+5u]));return aT(P,b4);}`
