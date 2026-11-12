// SPDX-License-Identifier: AGPL-3.0-or-later
// Numeric data never passes through an image decoder, color profile or alpha conversion.
(function(root) {
'use strict';
async function decode(buffer) {
  const bytes = new Uint8Array(buffer);
  if(bytes.length < 12 || new TextDecoder().decode(bytes.slice(0,8)) !== 'LWXRT01\n') throw Error('Unknown data tile');
  const length = new DataView(bytes.buffer, bytes.byteOffset).getUint32(8,true);
  if(length < 1 || length > 4096 || length+12 > bytes.length) throw Error('Invalid header');
  const m = JSON.parse(new TextDecoder().decode(bytes.slice(12,12+length)));
  if(m.version !== 1 || ![m.size,m.side,m.pad,m.decoded_bytes].every(Number.isInteger) || m.size < 1 || m.size > 512 || m.pad < 0 || m.side !== m.size+2*m.pad || m.side > 2048 || typeof m.empty !== 'boolean' || typeof m.snow !== 'boolean' || (m.empty && m.snow) || typeof m.blur !== 'number' || !Number.isFinite(m.blur) || m.blur < 0 || m.blur > 128 || (m.blur < .5 && m.pad !== 0)) throw Error('Invalid metadata');
  if(m.intensity !== 'uint8: dbz=value/2-32; zero=no-echo' || m.snow_encoding !== 'little-bit-first') throw Error('Unknown encoding');
  const k=m.kernel;
  if(m.blur < .5 ? k !== null : (!k || ![k.radius,k.center,k.edge].every(Number.isInteger) || k.radius < 0 || k.radius > 256 || k.center < 0 || k.edge < 0 || k.passes !== 3 || k.scale !== 16777216 || Math.abs((2*k.radius+1)*k.center+2*k.edge-k.scale)>1)) throw Error('Invalid blur kernel');
  const n=m.side*m.side, expected=m.empty ? 0 : n+(m.snow ? Math.ceil(n/8):0);
  if(m.decoded_bytes !== expected) throw Error('Invalid plane sizes');
  const reader=new Blob([bytes.slice(12+length)]).stream().pipeThrough(new DecompressionStream('deflate')).getReader();
  const raw=new Uint8Array(expected); let offset=0;
  try { while(true) { const {done,value}=await reader.read(); if(done)break; if(offset+value.length>expected) throw Error('Inflated payload exceeds limit'); raw.set(value,offset); offset+=value.length; } }
  catch(error) { await reader.cancel().catch(()=>{}); throw error; }
  if(offset!==expected) throw Error('Truncated planes');
  return {metadata:m,values:raw.subarray(0,n),snow:m.snow ? raw.subarray(n):null};
}
function pass(input, side, kernel, vertical) {
  const out=new Uint8Array(input.length), r=kernel.radius, scale=kernel.scale;
  const prefix=new Float64Array(side+1);
  for(let line=0;line<side;line++) for(let channel=0;channel<4;channel++) {
    const index=p=> ((vertical ? p*side+line : line*side+p)*4+channel);
    prefix[0]=0; for(let p=0;p<side;p++)prefix[p+1]=prefix[p]+input[index(p)];
    for(let p=0;p<side;p++) {
      const lo=p-r,hi=p+r;
      const sum=prefix[Math.min(side,hi+1)]-prefix[Math.max(0,lo)] + Math.max(0,-lo)*input[index(0)] + Math.max(0,hi-side+1)*input[index(side-1)];
      const far=input[index(Math.max(0,lo-1))]+input[index(Math.min(side-1,hi+1))];
      out[index(p)]=Math.floor((sum*kernel.center+far*kernel.edge+scale/2)/scale);
    }
  }
  return out;
}
function colorize(tile,palette) {
  const m=tile.metadata, n=m.side*m.side;
  if(m.empty)return new Uint8Array(m.size*m.size*4);
  for(const lut of [palette.rain,palette.snow]) if(!Array.isArray(lut) || lut.length!==256 || lut.some(row=>!Array.isArray(row)||row.length!==4||row.some(v=>!Number.isInteger(v)||v<0||v>255)))throw Error('Invalid palette');
  let rgba=new Uint8Array(n*4);
  for(let p=0;p<n;p++) { const snow=tile.snow && ((tile.snow[p>>3]>>(p&7))&1); rgba.set((snow?palette.snow:palette.rain)[tile.values[p]],p*4); }
  if(m.kernel) for(const vertical of [false,true]) for(let i=0;i<3;i++)rgba=pass(rgba,m.side,m.kernel,vertical);
  if(!m.pad)return rgba;
  const cropped=new Uint8Array(m.size*m.size*4);
  for(let y=0;y<m.size;y++)cropped.set(rgba.subarray(((y+m.pad)*m.side+m.pad)*4,((y+m.pad)*m.side+m.pad+m.size)*4),y*m.size*4);
  return cropped;
}
const api={decode,colorize};
if(typeof module!=='undefined')module.exports=api; else root.RadarData=api;
})(globalThis);
