// H4 planogram engine core — Rust. Bit-exact with C/Python (same deterministic input).
const BIAS: u32 = 512;
#[inline] fn h4_encode(b:&[u8;4])->[u16;4]{
    let (x,y,w,h)=(b[0] as u32,b[1] as u32,b[2] as u32,b[3] as u32);
    [(x+y+w+h) as u16, ((x+w+BIAS)-(y+h)) as u16, ((x+y+BIAS)-(w+h)) as u16, ((x+h+BIAS)-(y+w)) as u16]
}
#[inline] fn h4_decode(e:&[u16;4])->[u8;4]{
    let (xx,yy,zz,ww)=(e[0] as u32,e[1] as u32,e[2] as u32,e[3] as u32);
    [((xx+yy+zz+ww-3*BIAS)>>2) as u8, (((xx+zz+BIAS)-(yy+ww))>>2) as u8,
     (((xx+yy+BIAS)-(zz+ww))>>2) as u8, (((xx+ww+BIAS)-(yy+zz))>>2) as u8]
}
fn main(){
    let n:usize = std::env::args().nth(1).and_then(|s|s.parse().ok()).unwrap_or(13184);
    let mut b=vec![0u8;n*4];
    for i in 0..n*4 { b[i]=((i*167+13)&0xFF) as u8; }
    let (mut errs,mut csum)=(0u64,0u64);
    for i in 0..n {
        let bx=[b[i*4],b[i*4+1],b[i*4+2],b[i*4+3]];
        let e=h4_encode(&bx); let d=h4_decode(&e);
        for j in 0..4 { if d[j]!=bx[j]{errs+=1;} csum=csum.wrapping_mul(131).wrapping_add(e[j] as u64); }
    }
    let t0=std::time::Instant::now(); let mut sink=0u64;
    for _ in 0..1000 { for i in 0..n { let bx=[b[i*4],b[i*4+1],b[i*4+2],b[i*4+3]]; sink=sink.wrapping_add(h4_encode(&bx)[0] as u64);} }
    let ns=t0.elapsed().as_nanos() as f64/(1000.0*n as f64);
    println!("Rust: N={}  roundtrip_errors={}  checksum={}  encode={:.4} ns/box  (sink={})",n,errs,csum,ns,sink&1);
}
